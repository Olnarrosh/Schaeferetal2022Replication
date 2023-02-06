import sklearnfile
import pickle

#script to run train pipeline
#only change parameters for corpus and model
# 20 trainobjects


# input: 90% testsplit of desired corpus, model as str
def train_model(corpus, model):
    #switch case for 4 models
    matrix = sklearnfile.create_matrix(corpus)
    vec = sklearnfile.create_decision_vec(corpus)

    match model:
        case "PyTorch":

        case "LogisticRegression":
            lr_trained = sklearnfile.train_lr(matrix, vec)
            filename = "lr_train.sav"
            pickle.dump(lr_trained, open(filename, "wb"))
        case "RandomForest":
            rf_trained = sklearnfile.train_rf(matrix, vec)
            filename = "rf_train.sav"
            pickle.dump(rf_trained, open(filename, "wb"))
        case "SVM":
            svm_trained = sklearnfile.train_svm(matrix, vec)
            filename = "lr_train.sav"
            pickle.dump(svm_trained, open(filename, "wb"))
    return #fertig trainiertes Modell als pickle


if __name__ == "__main__":
