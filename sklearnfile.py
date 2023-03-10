"""
@Author Tana Deeg
"""
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np
import preprocess
import parse_cmv
import pickle



# create matrix (n_samples, n_features) by extracting embeddings from list of tuples (embedding, class)
def create_matrix(corpuslist):
    matrix = []
    for i in corpuslist:
        matrix.append(i[1])
    return np.array(matrix)

# create vector by extracting the value class (0 or 1) from list of tuples (string, embedding, class)
def create_decision_vec(corpuslist):
    vector = []
    for i in corpuslist:
        vector.append(i[2])
    return vector


# merge matrix and vector function to have input for models in one place
def create_sklearn_input(corpuslist):
    return create_matrix(corpuslist), create_decision_vec(corpuslist)



#trains a classifier using logistic regression
#input: corpuslist of format [(str, embedding, bool),...]
def train_lr(corpuslist):
    matrix, vector = create_sklearn_input(corpuslist)
    clf_lr = LogisticRegression(random_state=0).fit(matrix, vector)
    return clf_lr

# predict with logisticRegression class and matrix you want to be classified as input
# returns np array that contains the predicted class labels (length = num of lines of input matrix)
def predict_lr(clf_lr, corpuslist):
    inputmatrix = create_matrix(corpuslist)
    pred_lr = clf_lr.predict(inputmatrix)
    return pred_lr



#trains a classifier using a random forest classifier
#input: corpuslist of format [(str, embedding, bool),...]
def train_rf(corpuslist):
    matrix, vector = create_sklearn_input(corpuslist)
    clf_rf = RandomForestClassifier(random_state=0).fit(matrix, vector)
    return clf_rf

# predict with random forest class and matrix you want to be classified as input
# returns np array that contains the predicted class labels (length = num of lines of input matrix)
def predict_rf(rf_mlp, corpuslist):
    inputmatrix = create_matrix(corpuslist)
    pred_rf = rf_mlp.predict(inputmatrix)
    return pred_rf



#trains a classifier using a support vector machine
#input: corpuslist of format [(str, embedding, bool),...]
def train_svm(corpuslist):
    matrix, vector = create_sklearn_input(corpuslist)
    clf_svm = svm.SVC().fit(matrix, vector)
    return clf_svm

# predict with svm class and matrix you want to be classified as input
# returns np array that contains the predicted class labels (length = num of lines of input matrix)
def predict_svm(clf_svm, corpuslist):
    inputmatrix = create_matrix(corpuslist)
    pred_svm = (clf_svm.predict(inputmatrix))
    return pred_svm



# Compute Precision, Recall and FScore
# input: two lists of same length (gold and pred) containing the class labels -> should fit output of predict method
def run_evaluation(true, pred):
    return precision_score(true, pred), recall_score(true, pred), f1_score(true, pred), accuracy_score(true, pred)




