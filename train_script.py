"""
@Author Tana Deeg and Sandro Weick
"""
import sklearnfile
from custom_dataset_pytorch import CustomEmbeddingDataset
import pytorch_model as mod
import torch
from torch import nn
import pickle


# script to run train and evaluation pipeline
# only change parameters for corpus and model
# input: corpus str must be either "corpus" or "without_corpus" (for leave-one-out approach)
def train_model(corpus: str, model: str):
    # switch case for 4 models
    match model:
        case "pytorch":
            # open pickle file with data
            with open(f"./processed_data_results/train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)

            pytorch_model = mod.train_model(train_list)

            torch.save(pytorch_model, "./processed_data_results/pytorch_model.pt")

        case "lr":
            # open pickle file with data
            with open(f"./processed_data_results/train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_lr = sklearnfile.train_lr(train_list)
            # save trained model as pickle
            filename = "./processed_data_results/lr_model.sav"
            pickle.dump(trained_lr, open(filename, "wb"))

        case "rf":
            # open pickle file with data
            with open(f"./processed_data_results/train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_rf = sklearnfile.train_rf(train_list)
            # save trained model as pickle
            filename = "./processed_data_results/rf_model.sav"
            pickle.dump(trained_rf, open(filename, "wb"))

        case "svm":
            # open pickle file with data
            with open(f"./processed_data_results/train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # train model
            trained_svm = sklearnfile.train_svm(train_list)
            # save trained model as pickle
            filename = "./processed_data_results/svm_model.sav"
            pickle.dump(trained_svm, open(filename, "wb"))


def eval_model(corpus: str, model: str, ):
    # open validation portion of corpus
    with open(f"./processed_data_results/val_{corpus}_file.pkl", "rb") as file:
        val_list = pickle.load(file)

    # open gold list of corpus
    with open(f"./processed_data_results/gold_{corpus}_list.pkl", "rb") as gold:
        gold_list = pickle.load(gold)

    # load models and get prediction lists
    if model == "pytorch":
        loaded_model = torch.load("./processed_data_results/pytorch_model.pt")
        loaded_model.eval()
        predlist = mod.make_predictions(val_list, loaded_model)
    else:
        loaded_model = pickle.load(open(f"./processed_data_results/{model}_model.sav", 'rb'))
        match model:
            case "lr":
                # evaluate
                predlist = sklearnfile.predict_lr(loaded_model, val_list)
            case "rf":
                # evaluate
                predlist = sklearnfile.predict_rf(loaded_model, val_list)
            case "svm":
                # evaluate
                predlist = sklearnfile.predict_svm(loaded_model, val_list)

    #run evaluation method to get precision, recall and Fscore
    precision, recall, f1_score, accuracy = sklearnfile.run_evaluation(gold_list, predlist)
    eval_dict = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy
        }
    print(f"Precision: {precision}, Recall: {recall}, F-Score: {f1_score}, Accuracy: {accuracy}")
    with open(f"./processed_data_results/{corpus}_{model}_evaluation.pkl", "wb") as file:
        pickle.dump(eval_dict, file)


if __name__ == "__main__":
    """
    # cmv - cmv (In-Domain)
    print(f"cmv: pytorch")
    train_model("cmv", "pytorch")
    eval_model("cmv", "pytorch")
    print(f"cmv: lr")
    train_model("cmv", "lr")
    eval_model("cmv", "lr")
    print(f"cmv: rf")
    train_model("cmv", "rf")
    eval_model("cmv", "rf")
    print(f"cmv: svm")
    train_model("cmv", "svm")
    eval_model("cmv", "svm")
    
	# essay - essay (In-Domain)
    print(f"essay: pytorch")
    train_model("essay", "pytorch")
    eval_model("essay", "pytorch")
    print(f"essay: lr")
    train_model("essay", "lr")
    eval_model("essay", "lr")
    print(f"essay: rf")
    train_model("essay", "rf")
    eval_model("essay", "rf")
    print(f"essay: svm")
    train_model("essay", "svm")
    eval_model("essay", "svm")
    
	# micro - micro (In-Domain)
    print(f"micro: pytorch")
    train_model("micro", "pytorch")
    eval_model("micro", "pytorch")
    print(f"micro: lr")
    train_model("micro", "lr")
    eval_model("micro", "lr")
    print(f"micr: rf")
    train_model("micro", "rf")
    eval_model("micro", "rf")
    print(f"micro: svm")
    train_model("micro", "svm")
    eval_model("micro", "svm")
    
	# usdeb - usdeb (In-Domain)
    print(f"usdeb: pytorch")
    train_model("usdeb", "pytorch")
    eval_model("usdeb", "pytorch")
    print(f"usdeb: lr")
    train_model("usdeb", "lr")
    eval_model("usdeb", "lr")
    print(f"usdeb: rf")
    train_model("usdeb", "rf")
    eval_model("usdeb", "rf")
    print(f"usdeb: svm")
    train_model("usdeb", "svm")
    eval_model("usdeb", "svm")
    """
	# usdeb - micro (cross-Domain)
    print(f"usdeb-micro: pytorch")
    train_model("usdeb", "pytorch")
    eval_model("micro", "pytorch")
    print(f"usdeb: lr")
    train_model("usdeb", "lr")
    eval_model("micro", "lr")
    print(f"usdeb: rf")
    train_model("usdeb", "rf")
    eval_model("micro", "rf")
    print(f"usdeb: svm")
    train_model("usdeb", "svm")
    eval_model("micro", "svm")

    # usdeb - essay (cross-Domain)
    print(f"usdeb-essay: pytorch")
    train_model("usdeb", "pytorch")
    eval_model("essay", "pytorch")
    print(f"usdeb: lr")
    train_model("usdeb", "lr")
    eval_model("essay", "lr")
    print(f"usdeb: rf")
    train_model("usdeb", "rf")
    eval_model("essay", "rf")
    print(f"usdeb: svm")
    train_model("usdeb", "svm")
    eval_model("essay", "svm")

    # usdeb - cmv (cross-Domain)
    print(f"usdeb-cmv: pytorch")
    train_model("usdeb", "pytorch")
    eval_model("cmv", "pytorch")
    print(f"usdeb: lr")
    train_model("usdeb", "lr")
    eval_model("cmv", "lr")
    print(f"usdeb: rf")
    train_model("usdeb", "rf")
    eval_model("cmv", "rf")
    print(f"usdeb: svm")
    train_model("usdeb", "svm")
    eval_model("cmv", "svm")

    # micro - usdeb (cross-Domain)
    print(f"micro-usdeb: pytorch")
    train_model("micro", "pytorch")
    eval_model("usdeb", "pytorch")
    print(f"micro: lr")
    train_model("micro", "lr")
    eval_model("usdeb", "lr")
    print(f"micro: rf")
    train_model("micro", "rf")
    eval_model("usdeb", "rf")
    print(f"micro: svm")
    train_model("micro", "svm")
    eval_model("usdeb", "svm")

    # micro - essay (cross-Domain)
    print(f"micro-essay: pytorch")
    train_model("micro", "pytorch")
    eval_model("essay", "pytorch")
    print(f"micro: lr")
    train_model("micro", "lr")
    eval_model("essay", "lr")
    print(f"micro: rf")
    train_model("micro", "rf")
    eval_model("essay", "rf")
    print(f"micro: svm")
    train_model("micro", "svm")
    eval_model("essay", "svm")

    # micro - cmv (cross-Domain)
    print(f"micro-cmv: pytorch")
    train_model("micro", "pytorch")
    eval_model("cmv", "pytorch")
    print(f"micro: lr")
    train_model("micro", "lr")
    eval_model("cmv", "lr")
    print(f"micro: rf")
    train_model("micro", "rf")
    eval_model("cmv", "rf")
    print(f"micro: svm")
    train_model("micro", "svm")
    eval_model("cmv", "svm")

    # essay - usdeb (cross-Domain)
    print(f"essay-usdeb: pytorch")
    train_model("essay", "pytorch")
    eval_model("usdeb", "pytorch")
    print(f"essay: lr")
    train_model("essay", "lr")
    eval_model("usdeb", "lr")
    print(f"essay: rf")
    train_model("essay", "rf")
    eval_model("usdeb", "rf")
    print(f"essay: svm")
    train_model("essay", "svm")
    eval_model("usdeb", "svm")

    # essay - micro (cross-Domain)
    print(f"essay-micro: pytorch")
    train_model("essay", "pytorch")
    eval_model("micro", "pytorch")
    print(f"essay: lr")
    train_model("essay", "lr")
    eval_model("micro", "lr")
    print(f"essay: rf")
    train_model("essay", "rf")
    eval_model("micro", "rf")
    print(f"essay: svm")
    train_model("essay", "svm")
    eval_model("micro", "svm")

    # essay - cmv (cross-Domain)
    print(f"essay-cmv: pytorch")
    train_model("essay", "pytorch")
    eval_model("cmv", "pytorch")
    print(f"essay: lr")
    train_model("essay", "lr")
    eval_model("cmv", "lr")
    print(f"essay: rf")
    train_model("essay", "rf")
    eval_model("cmv", "rf")
    print(f"essay: svm")
    train_model("essay", "svm")
    eval_model("cmv", "svm")

    # cmv - usdeb (cross-Domain)
    print(f"cmv-usdeb: pytorch")
    train_model("cmv", "pytorch")
    eval_model("usdeb", "pytorch")
    print(f"cmv: lr")
    train_model("cmv", "lr")
    eval_model("usdeb", "lr")
    print(f"cmv: rf")
    train_model("cmv", "rf")
    eval_model("usdeb", "rf")
    print(f"cmv: svm")
    train_model("cmv", "svm")
    eval_model("usdeb", "svm")

    # cmv - micro (cross-Domain)
    print(f"cmv-micro: pytorch")
    train_model("cmv", "pytorch")
    eval_model("micro", "pytorch")
    print(f"cmv: lr")
    train_model("cmv", "lr")
    eval_model("micro", "lr")
    print(f"cmv: rf")
    train_model("cmv", "rf")
    eval_model("micro", "rf")
    print(f"cmv: svm")
    train_model("cmv", "svm")
    eval_model("micro", "svm")

    # cmv - essay (cross-Domain)
    print(f"cmv-essay: pytorch")
    train_model("cmv", "pytorch")
    eval_model("essay", "pytorch")
    print(f"cmv: lr")
    train_model("cmv", "lr")
    eval_model("essay", "lr")
    print(f"cmv: rf")
    train_model("cmv", "rf")
    eval_model("essay", "rf")
    print(f"cmv: svm")
    train_model("cmv", "svm")
    eval_model("essay", "svm")