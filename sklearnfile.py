import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
import corpus2embeddings
import parse_cmv



# create matrix (n_samples, n_features) by extracting embeddings from list of tuples (embedding, class)
def create_matrix(corpuslist):
    matrix = []
    for i in corpuslist:
        matrix.append(i[0])
    return np.array(matrix)

# create vector by extracting the value class (0 or 1) from list of tuples (string, embedding, class)
def create_decision_vec(corpuslist):
    vector = []
    for i in corpuslist:
        vector.append(i[1])
    return vector

def train(matrix, vector):
    clf = LogisticRegression(random_state=0).fit(matrix, vector)
    return clf

# predict with logisticRegression class and matrix you want to be classified as input
def predict(clf, inputmatrix):
    pred = clf.predict(inputmatrix)
    return pred

if __name__ == "__main__":
    #testdatei = [("bla", [1, 2, 32523, 423], 0), ("bla", [1235, 223562, 32234523, 42233], 1), ("blsdfa", [84351, 2, 325423, 3], 1), ("blasfa", [1, 2, 32523, 423], 0)]
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    tuplelist = corpus2embeddings.convert_corpus(loadedcorpus)

    testmatrix = create_matrix(tuplelist)
    testvec = create_decision_vec(tuplelist)
    traindone = train(testmatrix, testvec)
    print("training done")
    #print(predict(traindone, [[124, 2412, 3, 9]]))
