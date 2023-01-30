import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
import corpus2embeddings
import parse_cmv



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





#trains a classifier using logistic regression
#input: matrix and vector relative to matrix
def train_lr(matrix, vector):
    clf_lr = LogisticRegression(random_state=0).fit(matrix, vector)
    return clf_lr

# predict with logisticRegression class and matrix you want to be classified as input
# returns np array that contains the predicted class labels (length = num of lines of input matrix)
def predict_lr(clf_lr, inputmatrix):
    pred_lr = clf_lr.predict(inputmatrix)
    return pred_lr







#trains a classifier using a random forest classifier
#input: matrix and vector relative to matrix
def train_rf(matrix, vector):
    clf_rf = RandomForestClassifier(random_state=0).fit(matrix, vector)
    return clf_rf

def predict_rf(rf_mlp, inputmatrix):
    pred_rf = rf_mlp.predict(inputmatrix)
    return pred_rf








#trains a classifier using a support vector machine
#input: matrix and vector relative to matrix
def train_svm(matrix, vector):
    clf_svm = svm.SVC().fit(matrix, vector)
    return clf_svm

def predict_svm(clf_svm, inputmatrix):
    pred_svm = (clf_svm.predict(inputmatrix))
    return pred_svm








# compute precision
# input: two lists of same length (gold and pred) containing the class labels -> should fit output of predict method
def computeprecision(true, pred):
    return precision_score(true, pred)

# compute recall
# input: two lists of same length (gold and pred) containing the class labels -> should fit output of predict method
def computerecall(true, pred):
    return recall_score(true, pred)


# compute fscore
# input: two lists of same length (gold and pred) containing the class labels -> should fit output of predict method
def computefscore(true, pred):
    return f1_score(true, pred)





if __name__ == "__main__":
    """
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    print("loaded corpus")
    tuplelist = corpus2embeddings.convert_corpus(loadedcorpus)
    print("created tuplelist")
    trainmatrix = create_matrix(tuplelist)
    print("testmatrix done")
    trainvec = create_decision_vec(tuplelist)
    print("testvec done")
    traindone = train_svm(trainmatrix, trainvec)
    print("trainingdone")
    print(type(traindone))
    """
    testdatei = [("bla", [1, 2, 32523, 423], 0), ("bla", [1235, 223562, 32234523, 42233], 1), ("blsdfa", [84351, 2, 325423, 3], 1), ("blasfa", [1, 2, 32523, 423], 0)]
    testmatrix = create_matrix(testdatei)
    testvec = create_decision_vec(testdatei)
    trainedobj = train_rf(testmatrix, testvec)
    print("training done")
    res = predict_rf(trainedobj, [[124, 2412, 3, 9], [0, 0, 1, 24], [234, 124, 523, 632154], [1, 2, 32523, 423]])
    print(res)
    print(computefscore([1, 0, 1, 0], res))

