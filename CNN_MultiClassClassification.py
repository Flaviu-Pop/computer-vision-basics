import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import copy
import numpy as np

from torchvision import transforms
from tqdm import tqdm


############################################# ----- LOADING THE DATA ----- #############################################
batch_size = 8     # We set the batch size

# We set the path of the dataset (CIFAR 10 Dataset) on the local disk
data_root = 'C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Multi Class Classification Datasets\CIFAR 10/data/cifar10'


# We define the basic transformations of the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# We define more image augmentations
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

# Load the training dataset (if not exists local then downloads it) with the basic transformations
dataset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform,
)

"""
# Load the training dataset, but now the augmented images
augmented_dataset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform_augmentation,
)


# We do the union of the training datasets
dataset = dataset.__add__(augmented_dataset)
"""


# Set a part of the training set as a validation set
train_size = 40000
val_size = 10000

assert train_size + val_size <= len(dataset), "Too many elements!"

train_set, val_set, _ = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])


# Load the test set (if not exists on local disk then downloads it) with the basic transformations
test_set = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform,
)

"""
# Load the test set, but now with the transformed images
augmented_test_set = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform_augmentation,
)


# We do the union of these test datasets
test_set = test_set.__add__(augmented_test_set)
"""


classes = ('plane', 'car', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck')


# Set the loaders corresponding to each set
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)


############################################ ----- THE ARCHITECTURE ----- ##############################################
class CNN_MultiClassClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.convLayer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, bias=False, dtype=torch.float64)
        self.maxPoolLayer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convLayer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, dtype=torch.float64)
        self.maxPoolLayer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattenLayer = nn.Flatten()

        self.linearLayer1 = nn.Linear(in_features=1152, out_features=128, dtype=torch.float64)
        self.linearLayer2 = nn.Linear(in_features=128, out_features=64, dtype=torch.float64)
        self.linearLayer3 = nn.Linear(in_features=64, out_features=10, dtype=torch.float64)
        self.activationOutput = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.type(torch.DoubleTensor)
        x = self.convLayer1(x)
        x = F.relu(x)
        x = self.maxPoolLayer1(x)

        x = x.type(torch.DoubleTensor)
        x = self.convLayer2(x)
        x = F.relu(x)
        x = self.maxPoolLayer2(x)

        x = self.flattenLayer(x)

        x = self.linearLayer1(x)
        x = F.relu(x)
        x = self.linearLayer2(x)
        x = F.relu(x)
        x = self.linearLayer3(x)
        x = self.activationOutput(x)


        return x


################################################## ----- TRAIN() ----- #################################################
def compute_accuracy(model, data_loader):
    # Computes the <model> accuracy, based on the data_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # Set device to GPU if available

    model = model.to(device)
    model.eval()

    total_correct = 0

    for input, label in tqdm(data_loader, leave=True):
        input = input.type(torch.DoubleTensor)
        label = label.type(torch.DoubleTensor)

        input, label = input.to(device), label.to(device)      # Set the data to GPU if available

        output = model(input.double())
        output_class = output.argmax(1)
        correct = (output_class == label)

        total_correct += correct.sum()

    print(f"\nCorrect Items: {total_correct} --- All Items: {len(data_loader.dataset)}")

    return total_correct/len(data_loader.dataset)


def train(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n ----- The training process -----")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # Set device to GPU if available

    # Set the model to GPU if available
    model = model.to(device)

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    # The Training Loop
    for epoch in tqdm(range(num_epochs), leave=True):
        start_time = time.perf_counter()
        print(f"\n\n\n Starting the Epoch: {epoch + 1}:")

        model.train()  # Set the model to training mode

        total_loss = 0

        for input, label in tqdm(train_loader, leave=True):
            input, label = input.to(device), label.to(device)     # Set the data to GPU if available

            # Forward and Backward Passes
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
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

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {duration: .3f} ===> Validation Accuracy = {validation_accuracy: .4f} ===> Best Accuracy = {best_accuracy: .4f} at the Epoch {best_epoch}\n")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    print(f"\n Computing the Test Accuracy for the Best Model")
    test_accuracy = compute_accuracy(model, test_loader)
    print(f"\nThe Test Accuracy of the Final Models is: {test_accuracy: .4f}")

    # Save the best model, based on the Accuracy of the Vadidation Set
    path_best_model = "../computer-vision-basics/cnn_multiclass_classification_cifar10.pth"
    torch.save(model, path_best_model)


############################################# ----- MAIN() ----- #######################################################
if __name__ == '__main__':
    start_time = time.perf_counter()

    cnn = CNN_MultiClassClassification()

    number_of_epochs = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=1e-3)

    train(cnn, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
          num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"\n\n\nTotal Time: {duration}")
