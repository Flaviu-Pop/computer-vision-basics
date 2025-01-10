import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
import time
import copy

from torchvision import transforms
from tqdm import tqdm


########################################### --- DATA PRE-PROCESSING --- ################################################
# Loading the datasets (training set and test set) --- from local disk
train_dataset_path = "C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Binary Classification CV Datasets\Cat Vs Dog\\training_set"
test_dataset_path = "C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Binary Classification CV Datasets\Cat Vs Dog\\test_set"


# Basics transformations of the images from training and test datasets, respectively
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((64,64))
])

# Load the datasets with the basics transformations
train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(test_dataset_path, transform=transform)


# We add more images for our datasets, by augmenting the existing images
transform_augmentation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((64,64)),
    transforms.ColorJitter(brightness=0.5, contrast= 0.5, saturation= 0.5, hue= 0.5),
    transforms.Grayscale(num_output_channels= 3),
    transforms.RandomHorizontalFlip(p= 0.75),
    transforms.RandomVerticalFlip(p= 0.75),
    transforms.RandomRotation(degrees=45)
])

# Load the (existed) datasets with more (augmentation) transformations
train_dataset_augmentated = torchvision.datasets.ImageFolder(train_dataset_path, transform=transform_augmentation)
test_dataset_augmentated = torchvision.datasets.ImageFolder(test_dataset_path, transform=transform_augmentation)


# We do the unions of the corresponding datasets, in fact we double the datasets
train_dataset = train_dataset.__add__(train_dataset_augmentated)
# test_dataset = test_dataset.__add__(test_dataset_augmentated)


# We split the training-set into the train-set and the validation-set
# First we set the sizes of these sets
train_set_size = 12000
val_set_size = 4000

# Now we do the splitting
train_set, val_set, _ = torch.utils.data.random_split(
    train_dataset, [train_set_size, val_set_size, len(train_dataset) - train_set_size - val_set_size])


# Set the loaders corresponding to training, validation and testing datasets, respectively
batch_size = 16

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


########################################## ----- THE ARCHITECTURE ----- ################################################
class CNN_BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.convlutionalLayer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, bias=False, dtype=torch.float64)
        self.maxPooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convlutionalLayer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, dtype=torch.float64)
        self.maxPooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattenLayer = nn.Flatten()

        self.linearLayer1 = nn.Linear(in_features=6272, out_features=128, dtype=torch.float64)
        self.linearLayer2 = nn.Linear(in_features=128, out_features=16, dtype=torch.float64)
        self.linearLayer3 = nn.Linear(in_features=16, out_features=1, dtype=torch.float64)


    def forward(self, x):
        x = x.type(torch.DoubleTensor)
        x = self.convlutionalLayer1(x)
        x = F.relu(x)
        x = self.maxPooling1(x)

        x = x.type(torch.DoubleTensor)
        x = self.convlutionalLayer2(x)
        x = F.relu(x)
        x = self.maxPooling2(x)

        x = self.flattenLayer(x)

        x = self.linearLayer1(x)
        x = F.relu(x)
        x = self.linearLayer2(x)
        x = F.relu(x)
        x = self.linearLayer3(x)
        x = F.sigmoid(x)


        return x


################################### ----- THE TRAINING PROCESS ----- ###################################################
# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_accuracy(model, data_loader):
    # Computes the <model>'s accuracy on the <data_loader> dataset

    model = model.to(device)     # Set the <model> to GPU if available

    model.eval()   # Set the model to evaluation mode

    total_correct = 0
    for inputs, labels in tqdm(data_loader, leave=False):
        inputs = inputs.type(torch.DoubleTensor)
        labels = labels.type(torch.DoubleTensor)

        inputs, labels = inputs.to(device), labels.to(device)      # Set the data to GPU if available

        outputs = model(inputs.double())
        outputs = torch.reshape(input=outputs, shape=[16])
        outputs_rounded = torch.round(outputs)
        correct = (outputs_rounded == labels)
        total_correct += correct.sum()

    print(f"Correct Items= {total_correct} ----- All Items = {len(data_loader.dataset)}")

    return total_correct/len(data_loader.dataset)


def train(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n ----- The Training Process ... -----")

    model = model.to(device)     # Set the model to GPU if available

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    # The training loop
    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()

        print(f"\n\n\n Starting the Epoch: {epoch + 1}:")

        model.train()     # Set the model in training mode

        total_loss = 0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs = inputs.type(torch.DoubleTensor)
            labels = labels.type(torch.DoubleTensor)
            inputs, labels = inputs.to(device), labels.to(device)     # Set the data to GPU if available

            # Forward and Backward Pass
            optimizer.zero_grad()
            outputs = model(inputs.double())
            outputs = torch.reshape(input=outputs, shape=[16])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss

        # Take the model('s weights) with the best accuracy based on the Validation Set
        print(f"\n Computing the Validation Accuracy for Epoch {epoch + 1}:")
        validation_accuracy = compute_accuracy(model, val_loader)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {duration: .3f} ===> Validation Accuracy = {validation_accuracy: .4f}  ===> Best Accuracy = {best_accuracy: .4f} at the Epoch {best_epoch}\n")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    print(f"\n Computing the Test Accuracy for the (best) model:")
    test_accuracy = compute_accuracy(model, test_loader)
    print(f"\nThe Test Accuracy of the Final Models is: {test_accuracy: .4f}")

    # Save the best model, based on the Accuracy given by the Vadidation Set
    path_best_model = "../computer-vision-basics/cnn_binary_classification_catdog.pth"
    torch.save(cnn, path_best_model)


############################################  MAIN()  #################################################################
if __name__ == '__main__':
    number_of_epochs = 2

    cnn = CNN_BinaryClassification()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=1e-3)

    train(cnn, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)
