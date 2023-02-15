import sklearnfile
from custom_dataset_pytorch import CustomEmbeddingDataset
import pytorch_model as mod
import torch
from torch import nn
import pickle


# script to run train and evaluation pipeline
# only change parameters for corpus and model
# 20 trainobjects

def train_model(corpus: str, model):
    # add matrix and vecotr
    # switch case for 4 models
    match model:
        case "PyTorch":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)

            pytorch_model = mod.train_model(train_list)

            torch.save(pytorch_model, "pytorch_model.pt")

        case "LogisticRegression":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_lr = sklearnfile.train_lr(train_list)
            # save trained model as pickle
            filename = "lr_model.sav"
            pickle.dump(trained_lr, open(filename, "wb"))

        case "RandomForest":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_rf = sklearnfile.train_rf(train_list)
            # save trained model as pickle
            filename = "rf_model.sav"
            pickle.dump(trained_rf, open(filename, "wb"))

        case "SVM":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_svm = sklearnfile.train_svm(train_list)
            # save trained model as pickle
            filename = "svm_model.sav"
            pickle.dump(trained_svm, open(filename, "wb"))


def eval_model(model: str, corpus: str):
    # open validation portion of corpus
    with open(f"val_{corpus}_file.pkl", "rb") as file:
        val_list = pickle.load(file)

    # open gold list of corpus
    with open(f"gold_{corpus}_list.pkl") as gold:
        gold_list = pickle.load(gold)

    # load models and get prediction lists
    # TODO -> get prediction list for pytorch

    if model == "pytorch":
        loaded_model = torch.load("pytorch_model.pt")
        loaded_model.eval()
        #predlist = fehlt noch
    else:
        loaded_model = pickle.load(open(f"{model}_model.sav", 'rb'))

        match model:
            case "LogisticRegression":
                # evaluate
                predlist = sklearnfile.predict_lr(loaded_model, val_list)
            case "RandomForest":
                # evaluate
                predlist = sklearnfile.predict_rf(loaded_model, val_list)
            case "SVM":
                # evaluate
                predlist = sklearnfile.predict_svm(loaded_model, val_list)

    #run evaluation method to get precision, recall and Fscore
    print(sklearnfile.run_evaluation(gold_list, predlist))
# TODO print out / save Evaluation results


if __name__ == "__main__":
    pass
