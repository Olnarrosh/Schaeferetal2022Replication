import torch
from torch.utils.data import DataLoader
from custom_dataset_pytorch import CustomEmbeddingDataset
from torch import nn

# dummy-Datensatz for testing purposes

data_train = [
	["Das ist ein Test Satz", [1,2,0,4,5,6,7,8,9], 1],
	["Dieses Model wird super", [0,0,1,0,0,0,0,0,0], 0],
	["Ich brauche noch einen dritten Satz", [40,10,0,41,34,89,356,8,154], 1]
]

data_test = [
	["Wow, noch mehr Sätze", [4,5,0,7,8,9,10,11,12], 1],
	["Test-Satz nummer 2", [0,15,1,1,3,4,3,8,0], 0],
	["Und Drittens", [1,6,0,4,5,6,7,8,9], 1]
]



# define NeuralNetwork class
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.input_size = 9
		self.output_size = 3
		#TODO still need to fix the in_features and out_features
		# integers for nn.Linear functions
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(self.input_size, self.output_size), 
			nn.ReLU(), 
			nn.Linear(self.output_size, self.output_size), 
			nn.ReLU(), 
			nn.Linear(self.output_size, 2)
		)
	
	def forward(self, x):
		logits = self.linear_relu_stack(x)
		return logits


# test NeuralNetwork
if __name__ == "__main__":
	train_dataset = CustomEmbeddingDataset(data_train)
	test_dataset = CustomEmbeddingDataset(data_test)

	# use DataLoaders to read in CustomDataset objects
	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

	# define Device that is used to compute
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	model = NeuralNetwork().to(device)
	print(model)

"""
	X = torch.rand(9, device=device)
	logits = model(X)
	pred_probab = nn.Softmax()(logits)
	y_pred = pred_probab.argmax(1)
	print(f"Predicted class: {y_pred}")
"""

