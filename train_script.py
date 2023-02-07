import sklearnfile
from custom_dataset_pytorch import CustomEmbeddingDataset
import pytorch_model as mod
import torch
from torch import nn
import pickle

#script to run train pipeline
#only change parameters for corpus and model
# 20 trainobjects

def train_model(corpus: str, model):
    #add matrix and vcotr
    #switch case for 4 models
    match model:
        case "PyTorch":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)

            # custom_dataset_pytorch read in data
            train_dataset = CustomEmbeddingDataset(train_list)

            # use DataLoader, don`t shuffle, data is "pre"-shuffled
            train_dataloader = mod.DataLoader(train_dataset)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} device")

            # get input and output sizes for model
            input_size = len(train_list[0][1])
            hidden_size = 256

            #create model
            model = mod.NeuralNetwork(input_size=input_size, hidden_size=hidden_size).to(device)
            print(model)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            # mod.train(train_dataloader, model, loss_fn, optimizer, device)

            epochs = 5
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                mod.train(train_dataloader, model, loss_fn, optimizer, device)
            print("Done!")

            torch.save(model, "pytorch_model.pt")


        case "LogisticRegression":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # create matrix and vector from corpusdata to put into train method
            lr_matrix = sklearnfile.create_matrix(train_list)
            lr_vec = sklearnfile.create_decision_vec(train_list)
            # train model
            trained_lr = sklearnfile.train_lr(lr_matrix, lr_vec)
            # save trained model as pickle
            filename = "lr_model.sav"
            pickle.dump(trained_lr, open(filename, "wb"))

        case "RandomForest":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # create matrix and vector from corpusdata to put into train method
            rf_matrix = sklearnfile.create_matrix(train_list)
            rf_vec = sklearnfile.create_decision_vec(train_list)
            # train model
            trained_rf = sklearnfile.train_rf(rf_matrix, rf_vec)
            # save trained model as pickle
            filename = "rf_model.sav"
            pickle.dump(trained_rf, open(filename, "wb"))

        case "SVM":
            # open pickle file with data
            with open(f"train_{corpus}_file.pkl", "rb") as file:
                train_list = pickle.load(file)
            # create matrix and vector from corpusdata to put into train method
            svm_matrix = sklearnfile.create_matrix(train_list)
            svm_vec = sklearnfile.create_decision_vec(train_list)
            # train model
            trained_svm = sklearnfile.train_svm(svm_matrix, svm_vec)
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

	# load (TODO and run) models 
	# TODO -> get prediction lists
	if model == "pytorch":
		loaded_model = torch.load("pytorch_model.pt")
		loaded_model.eval()
	else:
		loaded_model = pickle.load(open(f"{model}_model.sav", 'rb'))
	# TODO run evaluation Method (sklearn precision, recall and f1-Score)
	# TODO print out / save Evaluation results


if __name__ == "__main__":
	pass