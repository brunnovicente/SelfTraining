from SelfTraining import StandardSelfTraining
from TriTraining import TriTraining
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle

datasets = {}

preprocessing_pipe = make_pipeline(
    #OneHotEncoder on "Sex" feature
    OneHotEncoder(categorical_features=[0], sparse=False),
    #Scale all from 0 to 1
    MinMaxScaler())
# Apply preprocessing pipe to dataset and store the dataset in dict.


dados = pd.read_csv('c:/basedados/sementes.csv')
dados = shuffle(dados).values

datasets["sementes"] = {
    "X": dados[:,0:-1],
    "y": dados[:,-1]
}

# Class that only holds a collection of different 
# base classifiers for usage with SSL methods.
class base_classifiers:
    KNN = KNeighborsClassifier(
        n_neighbors=3,
        metric="euclidean",
        #n_jobs=2  # Parallelize work on CPUs
    )
    NB = GaussianNB(
        priors=None
    )
    #SVM = SVC(
    #    C=1.0,
    #    kernel='poly',
    #    degree=1,
    #    tol=0.001,
    #)
    CART = DecisionTreeClassifier(
        criterion='entropy',
        # splitter='best',
        # max_depth=None,
        # min_samples_split=2,
        min_samples_leaf=2,
        # min_weight_fraction_leaf=0.0,
        # max_features=None,
        # random_state=None,
        # max_leaf_nodes=None,
        # min_impurity_split=1e-07,
        # class_weight=None,
        # presort=False,
    )

# All classifiers used for testing
classifiers = [
    TriTraining("TriTraining (KNN)", base_classifiers.KNN),
    TriTraining("TriTraining (NB)", base_classifiers.NB),
    #TriTraining("TriTraining (SVM)", base_classifiers.SVM),
    TriTraining("TriTraining (CART)", base_classifiers.CART),
    StandardSelfTraining("Self-Training (KNN)", base_classifiers.KNN),
    StandardSelfTraining("Self-Training (NB)", base_classifiers.NB),
    #StandardSelfTraining("Self-Training (SVM)", base_classifiers.SVM),
    StandardSelfTraining("Self-Training (CART)", base_classifiers.CART)
]
labeling_rates = [0.10, 0.20, 0.30, 0.40]

def _training_scoring_iteration(clf, X, y, training_index, test_index, labeling_rate):
    """ 
    One iteration of fully training and scoring a 
    classifier on given data (one Kfold split)
    """
    #Testing set is set aside.. - 1/10th of the data
    X_test, y_test = X[test_index], y[test_index]

    #For generating a testing and transductive set
    split_data = train_test_split(
        X[training_index],
        y[training_index],
        test_size=labeling_rate,
        random_state=42
    )
    (X_unlabeled, X_labeled, y_unlabeled, y_labeled) = split_data

    #Training set - 9/10 of data
    X_train = np.concatenate((X_labeled, X_unlabeled))
    y_train = np.concatenate((
        y_labeled.astype(str),
        np.full_like(y_unlabeled.astype(str), "unlabeled")
    ))
    
    #Train the classifier
    clf.fit(X_train, y_train)
    
    #Score the classifier
    transductive_score = clf.score(X_unlabeled, y_unlabeled.astype(str))
    testing_score = clf.score(X_test, y_test.astype(str))

    cnf_matrix = pd.DataFrame(
        confusion_matrix(y_test.astype(str), clf.predict(X_test).astype(str))
    )
    
    return transductive_score, testing_score, cnf_matrix
    
def train_and_score(clf, X, y, cv, labeling_rate):
    """
    Perform KFold cross-validation of a classifier on a given data
    and labelling rate
    """
    transductive_scores = []
    testing_scores = []
    for training_index, test_index in cv.split(X,y):
        transductive_score, testing_score, cnf_matrix = _training_scoring_iteration(clf, X, y, training_index, test_index, labeling_rate)
        
        transductive_scores.append(transductive_score)
        testing_scores.append(testing_score)
        print("#", end="")
    print()
    scores = {
        "trans_mean": np.mean(transductive_scores),
        "test_mean": np.mean(testing_scores),
        "trans_std": np.std(transductive_scores),
        "test_std": np.std(testing_scores)
    }
    return scores, cnf_matrix


""" 
The main loop for testing 
all classifiers with 
all datasets and 
all labeling rates
"""
results = None
cnf_matrixes = {}
for classifier in classifiers:
    cnf_matrixes[classifier.name] = {}
    print(classifier.name)
    for dataset_name, dataset in datasets.items():
        cnf_matrixes[classifier.name][dataset_name] = {}
        print("dataset:", dataset_name, "\t")
        for labeling_rate in labeling_rates:
            print("rate:", labeling_rate, end=" ")

            test_info = { "classifier": classifier.name, "dataset":dataset_name, "labeling_rate":labeling_rate}
            cv = KFold(n_splits=10, random_state=42)
            scores, cnf_matrix = train_and_score(classifier, dataset["X"], dataset["y"], cv, labeling_rate)

            if results is None:
                results = pd.DataFrame([{**test_info, **scores}])
            else:
                results.loc[len(results.index)] = {**test_info, **scores}
            cnf_matrixes[classifier.name][dataset_name][labeling_rate] = cnf_matrix
    print()
    print("--------")