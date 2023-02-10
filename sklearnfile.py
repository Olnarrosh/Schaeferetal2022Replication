import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
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
def predict_lr(clf_lr, inputmatrix):
    pred_lr = clf_lr.predict(inputmatrix)
    return pred_lr







#trains a classifier using a random forest classifier
#input: corpuslist of format [(str, embedding, bool),...]
def train_rf(corpuslist):
    matrix, vector = create_sklearn_input(corpuslist)
    clf_rf = RandomForestClassifier(random_state=0).fit(matrix, vector)
    return clf_rf

def predict_rf(rf_mlp, inputmatrix):
    pred_rf = rf_mlp.predict(inputmatrix)
    return pred_rf








#trains a classifier using a support vector machine
#input: corpuslist of format [(str, embedding, bool),...]
def train_svm(corpuslist):
    matrix, vector = create_sklearn_input(corpuslist)
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


# compute accuracy
# input: two lists of same length (gold and pred) containing the class labels -> should fit output of predict method
def computeaccuracy(true, pred):
    return accuracy_score(true, pred)





if __name__ == "__main__":
    """
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    print("loaded corpus")
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    print(len(tuplelist))
    print("created tuplelist")
    trainmatrix = create_matrix(tuplelist)
    print("testmatrix done")
    trainvec = create_decision_vec(tuplelist)
    print("testvec done")
    traindone = train_svm(trainmatrix, trainvec)
    print("trainingdone")
    filename = "testfile.sav"
    pickle.dump(traindone, open(filename, "wb"))
    print(pickle.load(open(filename)))
    #print(type(traindone))
    #predictone = tuplelist[0][1]
    #print(predict_svm(traindone, [predictone]))
    """
    testdatei = [("bla", [1.0, 2.0, 32523.0, 423.0], 0), ("bla", [1235.0, 223562.0, 32234523.0, 42233.0], 1), ("blsdfa", [84351.0, 2.0, 325423.0, 3.0], 1), ("blasfa", [1.0, 2.0, 32523.0, 423.0], 0)]
    #testmatrix = create_matrix(testdatei)
    #testvec = create_decision_vec(testdatei)
    trainedobj = train_lr(testdatei)
    filename = "testfile.sav"
    print("training done")
    pickle.dump(trainedobj, open(filename, "wb"))
    #this is important!! how to open pickle again!!
    entpickled = pickle.load(open("testfile.sav", "rb"))
    #print(type(trainedobj))
    res = predict_lr(entpickled, [[124.0, 2412.0, 3.0, 9.0], [0.0, 0.0, 1.0, 24.0], [234.0, 124.0, 523.0, 632154.0], [1.0, 2.0, 32523.0, 423.0]])
    print(res)
    print(computefscore([1, 0, 1, 0], res), computeaccuracy([1, 0, 1, 1], res))
    """
    #Test cmv vorverarbeitung
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    print("loaded corpus")
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    print(len(tuplelist))
    """


