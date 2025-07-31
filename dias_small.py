#Importing important libraries
import torch
import os
from torch import nn
import numpy as np
import torch.distributed
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import copy
import sys
import layers   # Present in the same directory as main.py

from torch.utils.data.distributed import DistributedSampler as DS

import typing as typ

import torch.distributed as dist	# Distributed Package for Communication etc from pytorch
import datetime
from torchvision.models import vgg19, vgg16, alexnet

import statistics

def dropped_model_class_define(model:nn.Module, paras:typ.List[str], first_imp_layer:typ.List[int], last_imp_layer:typ.List[int])->None:
	global DroppedModel
	
	'''
	Function to define class for small model, DroppedModel will be in global scope
	model: large model
	paras: parameters arguement of sequential layers name
	first_imp_layer: list storing index for first imp layer in one sequetial layer of model. Imp means we can apply dropout on that layer.
	last_imp_layer: list storing index for last imp layer in one sequetial layer of model. Imp means we can apply dropout on that layer.
	'''
	#Build class for Dropped model
	class DroppedModel(nn.Module):
		def __init__(self, dropout = 0.5):
			'''
			dropout: how much neurons to drop per layer
			'''
			super(DroppedModel, self).__init__()			
			layer = []
			for idx, para in enumerate(paras):
				tlayers = []
				layer_count = 0
				if '__iter__' in dir(getattr(model, para)):
					for i, val in enumerate(getattr(model, para)):
						if type(val)== torch.nn.modules.linear.Linear:
							if last_imp_layer[idx]==i:
								if layer_count%2==0:
									tlayers.append(nn.Linear(val.in_features, val.out_features, bias='bias' in val.state_dict()))
								else:
									tlayers.append(nn.Linear(int(val.in_features*dropout), val.out_features, bias='bias' in val.state_dict()))
							elif layer_count%2==0:
								tlayers.append(nn.Linear(val.in_features, int(val.out_features*dropout), bias='bias' in val.state_dict()))
							else:
								tlayers.append(nn.Linear(int(val.in_features*dropout), val.out_features, bias='bias' in val.state_dict()))
							layer_count += 1

						elif type(val)== torch.nn.modules.conv.Conv2d:
							if last_imp_layer[idx]==i:
								if layer_count%2==0:
									tlayers.append(nn.Conv2d(val.in_channels, val.out_channels, kernel_size=val.kernel_size, stride= val.stride, padding=val.padding))
								else:
									tlayers.append(nn.Conv2d(int(val.in_channels*dropout), val.out_channels, kernel_size=val.kernel_size, stride= val.stride, padding=val.padding))
							elif layer_count%2==0:
								tlayers.append(nn.Conv2d(val.in_channels, int(val.out_channels*dropout), kernel_size=val.kernel_size, stride= val.stride, padding=val.padding))
							else:
								tlayers.append(nn.Conv2d(int(val.in_channels*dropout), val.out_channels, kernel_size=val.kernel_size, stride= val.stride, padding=val.padding))
							layer_count += 1

						else:
							tlayers.append(copy.copy(val))
					layer.append(tlayers)
				else:
					layer.append(copy.copy(getattr(model, para)))			
			for i, para in enumerate(paras):
				if '__iter__' in dir(getattr(model, para)):
					setattr(self, para, nn.Sequential(*layer[i]))
				else:
					setattr(self, para, getattr(model, para))
		def forward(self, input):
			out = input
			for i, para in enumerate(paras):
				out = getattr(self, para)(out)
			return out


def first_last_imp_layer(tmodel:nn.Module, paras:typ.Tuple[str]) -> typ.Tuple[typ.List[int], typ.List[int], int]:
	'''
	function to give first and last important layer indexes, and number of important layers within multiple sequential layers
	tmodel: model
	paras: parameters arguement of sequential layers name
	returns: first_imp_layer List[int], last_imp_layer List[int], cnt_layers_imp int
	'''
	first_imp_layer = []	#used for differentiating whether previous layer has dropout or not (layers at boundaries of sequential there dropout is not performed)
	last_imp_layer = []	 #used for differentiating whether this layer has dropout or not
	cnt_layers_imp = 0  #used while send/recv for tagging msgs
	for para in paras:
		is_first_in_para = True
		first_imp_layer_idx = None
		last_imp_layer_idx = None		
		if '__iter__' in dir(getattr(tmodel, para)):
			for idx, i in enumerate(getattr(tmodel, para)):
				if type(i)== torch.nn.modules.linear.Linear:					
					if is_first_in_para:
						first_imp_layer_idx = idx
						is_first_in_para = False
					last_imp_layer_idx = idx
					cnt_layers_imp += 1
				elif type(i)== torch.nn.modules.conv.Conv2d:					
					if is_first_in_para:
						first_imp_layer_idx = idx
						is_first_in_para = False
					last_imp_layer_idx = idx
					cnt_layers_imp += 1
		first_imp_layer.append(first_imp_layer_idx)
		last_imp_layer.append(last_imp_layer_idx)
	return first_imp_layer, last_imp_layer, cnt_layers_imp
class DropoutInfo:
	'''
	Class for storing dropout information of neurons
	'''
	def __init__(self, model:nn.Module, paras:typ.List[str], first_imp_layer:typ.List[int], last_imp_layer:typ.List[int])->None:
		'''
		model: ARD model which have dropout for weights
		paras: parameters arguement of sequential layers name. List of strings
		first_imp_layer: list storing index for first imp layer in one sequetial layer of model. Imp means we can apply dropout on that layer.
		last_imp_layer: list storing index for last imp layer in one sequetial layer of model. Imp means we can apply dropout on that layer.

		returns: None
		'''
		self.model = model #ARD model
		self.paras = paras
		self.first_imp_layer = first_imp_layer
		self.last_imp_layer = last_imp_layer	

	def calc_dropouts(self, division_strong_set=(1/3))->None:
		'''
		division_strong_set: float value between 0 and 1, representing percentage of neurons of a layer which will be present in strong set, and accordingly rest in weak set
								default 1/3. You may need to change this value for differnt model, dataset, and GPU devices, making sure that division is such that strong
								neurons are able accomadate in GPU device memory.

		calculate dropouts for neurons from dropout for weights by averaging all weight for a neuron

		returns: None
		'''
		self.dropouts_for_neurons = []	  #store dropouts value for neurons. List[List[tensors.float]]
		self.low_dropped = []			   #store index for neurons having low dropout value (strong neurons)(currently taken as 1/3rd). List[List[tensors.int]]
		self.two_third_high = []			#store index for neurons having high dropout value (weak neurons)(currently taken as 2/3rd). List[List[tensors.int]]

		if division_strong_set<0 or division_strong_set>=1:
			print('Number of neurons in strong set percentage should be between 0 and 1')
			sys.exit()

		for i in self.paras:

			#temporary variables
			temp_dropouts_for_neurons = []
			temp_low_dropped = []
			temp_two_third_high = []
			if '__iter__' in dir(getattr(self.model, i)):   # To check if layer iterable i.e. layer is Sequential then if true
				for j in getattr(self.model, i):
					if 'log_sigma2' in j.state_dict().keys():   # To check if layer is like Conv, Linear where dropout for weights learned
						if type(j)== layers.LinearARD:
							temp2 = torch.exp( torch.mean(j.log_alpha,1) )  #from log_alpha for weights get alpha for neurons
							final_dropouts = temp2 / (1 + temp2)	#from alpha for neurons get dropouts for neurons
							final_dropouts.to('cpu')
							temp_dropouts_for_neurons.append(final_dropouts)

							# Dividing in strong and weak sets
							idx = torch.argsort(final_dropouts)
							bound, lowerbound = self.return_bound(final_dropouts, idx)
							print(f"\t\t{bound}")
							temp_low_dropped.append(idx[:bound+1]) #good neurons
							temp_two_third_high.append(idx[bound+1:])
						
						elif type(j)== layers.Conv2dARD:
							temp2 = torch.exp( torch.mean(j.log_alpha,(1,2,3)) )	#from log_alpha for weights get alpha for neurons
							final_dropouts = temp2 / (1 + temp2)					#from alpha for neurons get dropouts for neurons
							final_dropouts.to('cpu')
							temp_dropouts_for_neurons.append(final_dropouts)

							# Dividing in strong and weak sets
							idx = torch.argsort(final_dropouts)
							bound, lowerbound = self.return_bound(final_dropouts, idx)
							print(f"\t\t{bound}")
							temp_low_dropped.append(idx[:bound+1]) #good neurons
							temp_two_third_high.append(idx[bound+1:])
					else:
						# If layer don't have learnable weights which can be dropped, then append None for them in corresponding list of dropouts, strong, weak set
						temp_dropouts_for_neurons.append(None)
						temp_low_dropped.append(None) 
						temp_two_third_high.append(None)
				
				self.dropouts_for_neurons.append(temp_dropouts_for_neurons)
				self.low_dropped.append(temp_low_dropped)
				self.two_third_high.append(temp_two_third_high)
			else:
				# If layer is not sequential, then append None for them (we assume that model conv, linear layers are within sequential layer)
				self.dropouts_for_neurons.append(None)
				self.low_dropped.append(None)
				self.two_third_high.append(None)

	def return_bound(self, final_dropouts:torch.Tensor, idx:torch.Tensor):
		final_dropouts_y = [final_dropouts[int(i.item())].detach().item() for i in idx]
		final_dropouts_x = [i for i in range(len(final_dropouts_y))]
		if final_dropouts.shape[0] >= 100:
			diff_y = [final_dropouts_y[i+1]-final_dropouts_y[i] for i in final_dropouts_x[:-1]]
			var = [statistics.variance(diff_y[:-i]) for i in range(1, len(diff_y) // 2)]
			min_ = torch.argsort(torch.Tensor(var))[:10]
			min_ = min_[1:] if min_[0] == 0 else min_
			new_diff = diff_y[:-min_[0]]
			max_ = len(new_diff)
			var = [statistics.variance(new_diff[i:]) for i in range(1, len(new_diff) // 4)]
			min_ = torch.argsort(torch.Tensor(var))[:10]
			min_ = min_[1:] if min_[0] == 0 else min_
			
			return min_[0], max_
		else:
			return int(final_dropouts.shape[0]*.20), int(final_dropouts.shape[0]*.80)

class SmallModelTrainer:
	'''
	class that takes care for generating small models
	'''
	def __init__(self, paras:typ.List[str], dropout:float=0.5, dropout_info:DropoutInfo=None)->None:
		'''
		Make list of tensors representing unmarked neurons/filters. initialize with all unmarked. only for weak set
		paras: parameters arguement of sequential layers name
		dropoutinfo: object of DropoutInfo (called by master process) or None (called by other processes)
		'''
		self.paras = paras
		self.dropout = dropout
		if dropout_info != None:
			#only if master process called then initialize neurons not visited
			self.neurons_not_visited = []
			
			for idx, para in enumerate(self.paras):
				if dropout_info.dropouts_for_neurons[idx]!=None:
					neurons_not_visited_temp = []
					
					for i in range(len(dropout_info.dropouts_for_neurons[idx])):
						if dropout_info.dropouts_for_neurons[idx][i] != None:
							neurons_not_visited_temp.append(set(dropout_info.two_third_high[idx][i].tolist()))
						else:
							neurons_not_visited_temp.append(None)
					self.neurons_not_visited.append(neurons_not_visited_temp)
				else:
					self.neurons_not_visited.append(None)
			#stores how many small models generated
			self.step_id = 1
	
	def refresh_init(self, paras:typ.List[str], dropout:float=0.5, dropout_info:DropoutInfo=None):
		if dropout_info != None:
			#only if master process called then initialize neurons not visited
			self.neurons_not_visited = []
			
			for idx, para in enumerate(self.paras):
				if dropout_info.dropouts_for_neurons[idx]!=None:
					neurons_not_visited_temp = []
					
					for i in range(len(dropout_info.dropouts_for_neurons[idx])):
						if dropout_info.dropouts_for_neurons[idx][i] != None:
							neurons_not_visited_temp.append(set(dropout_info.two_third_high[idx][i].tolist()))
						else:
							neurons_not_visited_temp.append(None)
					self.neurons_not_visited.append(neurons_not_visited_temp)
				else:
					self.neurons_not_visited.append(None)

	def init_indexes(self, dropout_info:DropoutInfo=None):
		self.indexes = []
		for i, val in enumerate(dropout_info.dropouts_for_neurons):
			temp_indicies = []
			if val != None:
				for j, val2 in enumerate(val):
					temp_temp_indicies = []
					temp_indicies.append(temp_temp_indicies)
			self.indexes.append(temp_indicies)

	def generate_small_model(self, dropout_info:DropoutInfo=None, large_model:nn.Module=None, src_rank:int=None, dst_rank:int=None, cnt_layers_imp:int=None, neurons_to_ignore:list=None, set_no:int=0)->None:
		'''
		get small model
		master process will make small model and initialize weights from large model in this function
		child process will make small model and initialize weights from weights send by master process in this function
		dropout_info: object of DropoutInfo (called by master process) or None (called by child processes)
		large_model: only set by master process

		src_rank: source rank process. for child process.
		dst_rank: rank of process which will receive. for child process
		cnt_layers_imp: number of imp layers (dropout can be applied). for child process
		'''
		self.dropped_model = DroppedModel(dropout=self.dropout)
		if dropout_info!=None:
			#only if master process
			if set_no == 0:
				self.init_indexes(dropout_info=dropout_info) #stores indexes of neurons taken in small model from large model, List[List[tensor]]
			self.mapping = {}   #stores mapping (index of sequential layer, index of layer within sequential layer) -> index for layer in indexes List
			
			for i, val in enumerate(dropout_info.dropouts_for_neurons):
				tindexes = []
				layer_count = 0
				if val != None:
					for j, val2 in enumerate(val):
						if val2!=None:
							self.mapping[(i,j)] = len(tindexes)
							if dropout_info.last_imp_layer[i]==j:
								num_hidden = int(val2.shape[0])
							elif layer_count%2==0:
								num_hidden = int(val2.shape[0]*self.dropout)
							else:
								num_hidden = int(val2.shape[0])
							#num_hidden = int(val2.shape[0]*self.dropout) if layer_count%2 == 0 else int(val2.shape[0])
							bad_int = num_hidden - dropout_info.low_dropped[i][j].shape[0]

							if dropout_info.last_imp_layer[i]!=j:
								#torch.no_grad() imp because updating parameters that requires gradient
								with torch.no_grad():
									if set_no == 0:
										#first time generating small model
										if num_hidden == int(val2.shape[0]):
											two_third = np.array(list(self.neurons_not_visited[i][j]))
										else:
											temp_prob = np.asarray(dropout_info.dropouts_for_neurons[i][j].to('cpu')[list(self.neurons_not_visited[i][j])].to('cpu') / \
															torch.sum(dropout_info.dropouts_for_neurons[i][j].to('cpu')[list(self.neurons_not_visited[i][j])], dim=0)).astype('float64')
											temp_prob = 1 - temp_prob   #probability for selection opposite of dropout (high dropout, less probability for selection)
											temp_prob = temp_prob / np.sum(temp_prob)
											two_third = np.random.choice(list(self.neurons_not_visited[i][j]), bad_int, replace=False)
										
										select_of_neurons = dropout_info.low_dropped[i][j].to('cpu').tolist()
										select_of_neurons.extend(two_third.tolist())
										select_of_neurons = torch.cat((dropout_info.low_dropped[i][j].to('cpu'), torch.Tensor(two_third)))

										self.indexes[i][j] = select_of_neurons

								tindexes.append(self.indexes[i][j])
								layer_count += 1

			for i, val in enumerate(dropout_info.dropouts_for_neurons):
				if val != None:
					last_non_zero_j = -1
					for j, val2 in enumerate(val):
						if val2!=None:
							#update small model weights to that of large model weights
							if type(getattr(large_model, self.paras[i])[j])== torch.nn.modules.linear.Linear:
								if dropout_info.first_imp_layer[i]==j:
									getattr(self.dropped_model, self.paras[i])[j].out_features = len(self.indexes[i][j])

									nW = getattr(large_model, self.paras[i])[j].weight[self.indexes[i][j].long(), :]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias[self.indexes[i][j].long()]
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
								elif dropout_info.last_imp_layer[i]==j:
									getattr(self.dropped_model, self.paras[i])[j].in_features = len(self.indexes[i][last_non_zero_j])

									nW = getattr(large_model, self.paras[i])[j].weight[:, self.indexes[i][last_non_zero_j].long()]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias.clone().detach()
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
								else:
									getattr(self.dropped_model, self.paras[i])[j].out_features = len(self.indexes[i][j])
									getattr(self.dropped_model, self.paras[i])[j].in_features = len(self.indexes[i][last_non_zero_j])

									nW = getattr(large_model, self.paras[i])[j].weight[self.indexes[i][j].long(), :][:, self.indexes[i][last_non_zero_j].long()]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias[self.indexes[i][j].long()]
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
									
							elif type(getattr(large_model, self.paras[i])[j])== torch.nn.modules.conv.Conv2d:
								if dropout_info.first_imp_layer[i]==j:
									getattr(self.dropped_model, self.paras[i])[j].out_channels = len(self.indexes[i][j])

									nW = getattr(large_model, self.paras[i])[j].weight[self.indexes[i][j].long(), :, :, :]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias[self.indexes[i][j].long()]
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
								elif dropout_info.last_imp_layer[i]==j:
									getattr(self.dropped_model, self.paras[i])[j].in_channels = len(self.indexes[i][last_non_zero_j])

									nW = getattr(large_model, self.paras[i])[j].weight[:, self.indexes[i][last_non_zero_j].long(), :, :]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias.clone().detach()
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
								else:
									getattr(self.dropped_model, self.paras[i])[j].out_channels = len(self.indexes[i][j])
									getattr(self.dropped_model, self.paras[i])[j].in_channels = len(self.indexes[i][last_non_zero_j])

									nW = getattr(large_model, self.paras[i])[j].weight[self.indexes[i][j].long(), :, :, :][:, self.indexes[i][last_non_zero_j].long(), :, :]
									with torch.no_grad():
										getattr(self.dropped_model, self.paras[i])[j].weight = nn.Parameter(nW)
									if getattr(large_model, self.paras[i])[j].bias!=None:
										nB = getattr(large_model, self.paras[i])[j].bias[self.indexes[i][j].long()]
										with torch.no_grad():
											getattr(self.dropped_model, self.paras[i])[j].bias = nn.Parameter(nB)
									last_non_zero_j = j
			self.step_id += 1
		
	


def get_large_model(model)->typ.Tuple[nn.Module, typ.List[str]]:
	'''
	Defines large model and parameters arguement of sequential layers name
	returns: model, list of strings
	'''
	
	paras = [k for k,m in model._modules.items()]
	model.classifier = nn.Sequential(nn.Flatten(), *list(model.classifier._modules.values()))
	return model, paras
def split_dataset_by_node(dataset, node_rank, num_nodes):
    total_len = len(dataset)
    shard_size = total_len // num_nodes
    start = node_rank * shard_size
    end = (start + shard_size) if node_rank < num_nodes - 1 else total_len
    return torch.utils.data.Subset(dataset, list(range(start, end)))

def get_node_id(rank):
    return rank // torch.cuda.device_count()

def get_node_indices(dataset, num_nodes, node_id):
    total_size = len(dataset)
    node_part_size = total_size // num_nodes
    start = node_id * node_part_size
    end = start + node_part_size
    return list(range(start, end))




class Trainer:
		'''Class for training and testing models; Primarily for small model training.'''
		def __init__(self, model:nn.Module, device:str, train_data:DataLoader, test_data:DataLoader, batches_percent:float=1.0, optimizer:torch.optim.Optimizer=None)->None:
				'''
				model: model which needs training testing
				device: where model will be trained can be 'cpu' or 'gpu'
				train_data: data loader for train dataset
				test_data: data loader for test dataset
				optimizer: optimezer to use while training
				
				returns: None
				'''
				self.model = model.to(device)
				self.train_data = train_data
				self.global_rank = int(os.environ['RANK']) #global rank of process across machine set by torchrun
				self.device = device
				self.test_data = test_data
				self.optimizer = optimizer
				self.epochs_trained = 0
				self.batches_percent = batches_percent

		def _run_epoch(self, epoch:int)->float:
				'''
				Function for training model for one epoch; Called within train()
				epoch: integer stores current epoch number

				returns: float
				'''
				self.model.train()
				size = len(self.train_data.dataset)
				correct = 0
				for batch, (X, y) in enumerate(self.train_data):
						X, y = X.to(self.device), y.to(self.device)
						self.optimizer.zero_grad()
						out = self.model(X)
						correct += (out.argmax(1) == y).type(torch.float).sum().item()####
						loss = nn.CrossEntropyLoss()(out, y)	#calculates loss
						loss.backward()						 #computes gradient
						self.optimizer.step()				   #update weights
						del X, y
						
						if batch == int(self.batches_percent*len(self.train_data)):
							break

				return correct/int(self.batches_percent*size)
										  
		def train(self, max_epochs:int, early_stop:bool=False, tolerance:int=3):
			'''
			Function for training model for max_epochs epochs
			max_epochs: integer stores number of epoch to train model

			returns: int
			'''
			self.model.to(self.device)

			if self.optimizer==None:
				print(self.device,self.global_rank, " Define optimizer for training") 
			else:
				accuracies = []
				for epoch in range(max_epochs):
					start = datetime.datetime.now()
					accuracy = self._run_epoch(epoch)
					print(f"\t\tProcess {self.global_rank} | train time: {datetime.datetime.now()-start} | accuracy: {accuracy}")
					if early_stop == True:
						if len(accuracies)==0:
							accuracies.append(accuracy)
						else:
							last = len(accuracies)-1
							if accuracies[last]*0.98 <= accuracy and accuracy <= accuracies[last]*1.02:
								accuracies.append(accuracy)
							else:
								accuracies=[accuracy]
						if len(accuracies)==tolerance:
							break
				print(f"\tSmall Model{self.global_rank}: Epochs {epoch}")
				return epoch, accuracy

		def test(self)->None:
				'''
				Function for testing the model 

				returns: None
				'''
				self.model.to(self.device)
				size = len(self.test_data.dataset)
				num_batches = len(self.test_data)
				self.model.eval()   # model ready for testing
				test_loss, correct = 0, 0
				with torch.no_grad():
						for X, y in self.test_data:
								X, y = X.to(self.device), y.to(self.device)
								out = self.model(X)
								test_loss += nn.CrossEntropyLoss()(out,y).item()
								correct += (out.argmax(1) == y).type(torch.float).sum().item()
								del X, y
								
				test_loss /= num_batches
				correct /= size
				print(self.device, self.global_rank, f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
				self.model.to('cpu')
				return correct

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main():
	rank = int(os.environ['RANK'])
	gpu_id = int(os.environ['LOCAL_RANK'])
	rank = int(os.environ['RANK'])
	world_size = int(os.environ['WORLD_SIZE'])
	device = "cuda:"+str(gpu_id)
	setup(rank, world_size)

	def prepare_traindataloader(dataset:Dataset, batch_size:int)->DataLoader:
		'''
		Prepares Train Dataloader with Train Dataset

		returns: Dataloader
		'''
		sampler = NodeDistributedSampler(dataset, num_nodes=int(os.environ['WORLD_SIZE'])//torch.cuda.device_count(), rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
		return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
	class NodeDistributedSampler(torch.utils.data.Sampler):
		def __init__(self, dataset, num_nodes, rank, world_size):
			self.rank = rank
			self.world_size = world_size
			self.node_id = get_node_id(rank)
			self.num_nodes = num_nodes
			self.dataset = dataset
			self.indices = get_node_indices(dataset, num_nodes, self.node_id)

		def __iter__(self):
			return iter(self.indices)

		def __len__(self):
			return len(self.indices)

	def prepare_traindataloader(dataset:Dataset, batch_size:int)->DataLoader:
			'''
			Prepares Train Dataloader with Train Dataset

			returns: Dataloader
			'''
			sampler = NodeDistributedSampler(dataset, num_nodes=int(os.environ['WORLD_SIZE'])//torch.cuda.device_count(), rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
			return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

	def prepare_testdataloader(dataset:Dataset, batch_size:int)->DataLoader:
			'''
			Prepares Test Dataloader with Test Dataset

			returns: Dataloader
			'''
			return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	test_transform = transforms.Compose([
		transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		
	])
	# transform = transforms.Compose([
	# 			transforms.Resize((256,256)),
	# 			transforms.CenterCrop(224),
	# 			transforms.ToTensor(),
	# 			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	# ])

	# test_transform =transforms.Compose([
	# 			transforms.Resize((256, 256)),
	# 			transforms.CenterCrop(224),
	# 			transforms.ToTensor(),
	# 			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	# ])
	train_dataset = datasets.CIFAR10(
		root='/scratch/pusunuru/data',
		train=True,
		download=False,
		transform=transform
	)

	test_dataset = datasets.CIFAR10(
		root='/scratch/pusunuru/data',
		train=False,
		download=False,
		transform=test_transform
	)
	model = vgg19(weights=None, num_classes=10).to(device)
	model, paras = get_large_model(model)
	first_imp_layer, last_imp_layer, cnt_layers_imp = first_last_imp_layer(tmodel=model, paras=paras)
	dropped_model_class_define(model=model, paras=paras, first_imp_layer=first_imp_layer, last_imp_layer=last_imp_layer)
	dropout_info = DropoutInfo(model=model, paras=paras, first_imp_layer=first_imp_layer, last_imp_layer=last_imp_layer)
	dropout_info.calc_dropouts(division_strong_set=1/3)
	smt = SmallModelTrainer(paras=paras, dropout=0.5, dropout_info=dropout_info)
	smt.generate_small_model(src_rank=0, dst_rank=rank, cnt_layers_imp=cnt_layers_imp)
	model = smt.dropped_model
	# model = DroppedModel(dropout=0.5)
	model = model.to(device)
	optimizer = torch.optim.SGD(smt.dropped_model.parameters(), lr=1e-2, momentum=0.9)
	gpu_id = int(os.environ['LOCAL_RANK'])
	devicegpu = 'cuda:'+str(gpu_id)
	train_dataloader, test_dataloader = prepare_traindataloader(train_dataset, batch_size=64), prepare_testdataloader(test_dataset, batch_size=128)
	smt_model_trainer = Trainer( model=smt.dropped_model, device=devicegpu, train_data=train_dataloader, test_data=test_dataloader, optimizer=optimizer)
	epoch_count, train_acc = smt_model_trainer.train(max_epochs=10, early_stop=True)
	
if __name__ == "__main__":
    main()
