import torch
import torchvision

from torchvision import transforms
from CNN_BinaryClassification import CNN_BinaryClassification


torch.set_printoptions(precision=5, sci_mode=False)


# We load the model
model_path = "../computer-vision-basics/cnn_binary_classification_catdog.pth"
cnn = torch.load(model_path)


transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((64,64))      # The same size as in  training
])


# We load the images for prediction/inference from local disk
img_1_maia = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Binary Classification CV Datasets\Cat Vs Dog\single_prediction\maia2023.jpg").type(torch.FloatTensor).unsqueeze(0)
img_2_dog = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Binary Classification CV Datasets\Cat Vs Dog\single_prediction\cat_or_dog_1.jpg").type(torch.FloatTensor).unsqueeze(0)
img_3_cat = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Binary Classification CV Datasets\Cat Vs Dog\single_prediction\cat_or_dog_2.jpg").type(torch.FloatTensor).unsqueeze(0)


# Apply transformations to the images
img_1_maia = transform(img_1_maia)
img_2_dog = transform(img_2_dog)
img_3_cat = transform(img_3_cat)


# Make prediction for the images
prediction_1 = cnn(img_1_maia)
prediction_2 = cnn(img_2_dog)
prediction_3 = cnn(img_3_cat)


# Print the results
print(f"Prediction is: {prediction_1.item()}. Trebuie sa fie DOG (Maia).")
print(f"Prediction is: {prediction_2.item()}. Trebuie sa fie DOG.")
print(f"Prediction is: {prediction_3.item()}. Trebuie sa fie CAT.")
