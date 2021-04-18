import os
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision import models
from sklearn.model_selection import KFold
import math

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()



# class SimpleConvNet(nn.Module):
#     # 构造方法里面定义可学习参数的层
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     # forward定义输入数据进行前向传播
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x
class SimpleConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
    super(SimpleConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
    self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
    self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
    self.maxpool = nn.MaxPool2d(2, 2)
    self.avgpool = nn.AvgPool2d(2, 2)
    self.globalavgpool = nn.AvgPool2d(8, 8)
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.dropout50 = nn.Dropout(0.5)
    self.dropout10 = nn.Dropout(0.1)
    self.fc = nn.Linear(256, 10)

  def forward(self, x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.bn1(F.relu(self.conv2(x)))
    x = self.maxpool(x)
    x = self.dropout10(x)
    x = self.bn2(F.relu(self.conv3(x)))
    x = self.bn2(F.relu(self.conv4(x)))
    x = self.avgpool(x)
    x = self.dropout10(x)
    x = self.bn3(F.relu(self.conv5(x)))
    x = self.bn3(F.relu(self.conv6(x)))
    x = self.globalavgpool(x)
    x = self.dropout50(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  
if __name__ == '__main__':
  
  # Configuration options
  k_folds = 5
  num_epochs = 100
  loss_function = nn.CrossEntropyLoss()
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  # Prepare CIFAR10 dataset by concatenating Train/Test part; we split later.
  dataset_train_part = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
  dataset_test_part = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
  #dataset = ConcatDataset([dataset_train_part, dataset_test_part])
  dataset = dataset_train_part

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=32, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=32, sampler=test_subsampler)
    
    # Init the neural network
    network = SimpleConvNet()
    network.apply(reset_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'E:/deep_learning/Mnist_cross_vaildation/model/CIFAR10-model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data

        # Generate outputs
        outputs = network(inputs)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      # Print accuracy
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('--------------------------------')
      results[fold] = 100.0 * (correct / total)
    
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')
