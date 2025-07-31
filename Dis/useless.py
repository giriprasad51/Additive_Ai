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
