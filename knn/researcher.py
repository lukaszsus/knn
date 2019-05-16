import datetime
import os
import re
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from plotter import OUTCOME_PATH
from utils import *
from dsloader import *

warnings.filterwarnings('ignore')


class Researcher:
    NUM_TESTS_PER_EXAMPLE = 100
    NUM_FOLDS_MIN = 2
    NUM_FOLDS_MAX = 9
    NUM_K_MIN = 1
    NUM_K_MAX = 20
    VOTING_METHODS = ["uniform", "distance", count_one_over_sqrt_order_dist]
    VOTING_METHODS_NAMES = ["uniform", "distance", "sqrt-order"]
    DISTANCE_MEHTODS = [Distance.MANHATTAN.value, Distance.EUCLIDEAN.value]
    DATASET_INDICES = {"iris": 0, "diabetes": 0, "glass": 1, "wine": 2}
    METRICS_COLUMN_LIST = ["dataset", "n-k", "voting", "distance", "n-folds",
                           "acc_mean", "prec_mean", "rec_mean", "f1_mean"]

    def __init__(self, standarization=True):
        self._accuracies = None
        self._precisions = None
        self._recalls = None
        self._f1_scores = None

        self._splitter = None
        self._metrics: pd.DataFrame = pd.DataFrame(columns=self.METRICS_COLUMN_LIST)
        self._outcomes_dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        self._standarization = standarization
        self._scaler = preprocessing.StandardScaler()

        # params for research
        self._loader = None
        self._n_folds = None
        self._k = None
        self._voting_index = None
        self._voting = None
        self._dist = None

        # auxiliary
        self._data = None
        self._target = None

    def do_research_for_datasets(self):
        dataset_loaders = self._load_datasets()
        for self._loader in dataset_loaders:
            self._do_research_for_ks()

    def _do_research_for_ks(self):
        for self._k in range(self.NUM_K_MIN, self.NUM_K_MAX + 1):
            self._do_research_for_voting()

    def _do_research_for_voting(self):
        for self._voting_index in range(len(self.VOTING_METHODS)):
            self._voting = self.VOTING_METHODS[self._voting_index]
            self._do_research_for_distance()

    def _do_research_for_distance(self):
        for self._dist in self.DISTANCE_MEHTODS:
            self._do_research_for_folds()

    def _do_research_for_folds(self):
        self._data, self._target = self._loader()

        if self._standarization:
            self._data = self._scaler.fit_transform(self._data)

        for self._n_folds in range(self.NUM_FOLDS_MIN, self.NUM_FOLDS_MAX + 1):
            self._do_research_for_n_samples()

    def _do_research_for_n_samples(self):
        self._splitter = StratifiedKFold(n_splits=self._n_folds, shuffle=True)
        self._target = translate_class_labels(self._target)
        self._make_n_samples()
        self._metrics_summary()
        self._save_to_file()

    def _load_datasets(self):
        datasets_loader = list()
        datasets_loader.append(load_iris)
        datasets_loader.append(load_diabetes)
        datasets_loader.append(load_glass)
        datasets_loader.append(load_wine)
        return datasets_loader

    def _do_crossval(self):
        self._splitter = StratifiedKFold(n_splits=self._n_folds, shuffle=True)
        split_set_generator = self._splitter.split(self._data, self._target)

        # trainning and testing
        y_pred = list()
        y_true = list()

        for train_indices, test_indices in split_set_generator:
            X_train = self._data[train_indices]
            Y_train = self._target[train_indices]

            knn = KNeighborsClassifier(n_neighbors=self._k,
                                       weights=self._voting,
                                       algorithm='brute',
                                       p=self._dist)
            knn.fit(X_train, Y_train)

            y_pred.extend(knn.predict(self._data[test_indices]))
            y_true.extend(self._target[test_indices])

        confusion = metrics.confusion_matrix(y_true, y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average=None)
        recall = metrics.recall_score(y_true, y_pred, average=None)
        f1_score = metrics.f1_score(y_true, y_pred, average=None)

        return {"confusion": confusion, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1_score": f1_score}

    def _make_n_samples(self):
        self._accuracies = list()
        self._precisions = list()
        self._recalls = list()
        self._f1_scores = list()

        for i in range(self.NUM_TESTS_PER_EXAMPLE):
            metrics = self._do_crossval()

            self._accuracies.append(metrics["accuracy"])
            self._precisions.append(metrics["precision"])
            self._recalls.append(metrics["recall"])
            self._f1_scores.append(metrics["f1_score"])

        self._accuracies = np.asarray(self._accuracies)
        self._precisions = np.asarray(self._precisions)
        self._recalls = np.asarray(self._recalls)
        self._f1_scores = np.asarray(self._f1_scores)

    def _metrics_summary(self):
        mean_acc = np.mean(self._accuracies)
        mean_prec = np.mean(self._precisions)
        mean_rec = np.mean(self._recalls)
        mean_f1 = np.mean(self._f1_scores)

        record = pd.DataFrame([[self.__get_name_from_loader(),
                                self._k,
                                self.VOTING_METHODS_NAMES[self._voting_index],
                                self._dist,
                                self._n_folds,
                                mean_acc,
                                mean_prec,
                                mean_rec,
                                mean_f1]], columns=self.METRICS_COLUMN_LIST)
        if self._metrics.empty:
            self._metrics = record
        else:
            self._metrics = pd.concat([self._metrics, record], ignore_index=True)

        print("Dataset: {0}".format(self.__get_name_from_loader()))
        print("Value of k: {0}".format(self._k))
        print("Voting: {0}".format(self.VOTING_METHODS_NAMES[self._voting_index]))
        print("Distance: {0}".format(self._dist))
        print("N-folds: {0}".format(self._n_folds))
        print("Accuracy:\nmean: {0}".format(mean_acc))
        print("Precision:\nmean: {0}".format(mean_prec))
        print("Recall:\nmean: {0}".format(mean_rec))
        print("F1 score:\nmean: {0}".format(mean_f1))
        print()

    def _save_to_file(self):
        file_name = "{}.csv".format(self.__get_name_from_loader())
        path = os.path.join(OUTCOME_PATH, self._outcomes_dir_name)
        create_dir_if_not_exists(path)
        path = os.path.join(path, file_name)
        self._metrics.to_csv(path, index=False)

    def __get_name_from_loader(self):
        name = self._loader.__name__
        name = re.search("_.*", name)
        name = name.group(0)[1:]
        return name
