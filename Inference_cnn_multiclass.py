import torch
import torchvision

from torchvision import transforms
from CNN_MultiClassClassification import CNN_MultiClassClassification


torch.set_printoptions(precision=5, sci_mode=False)


# We load the model
model_path = "../computer-vision-basics/cnn_multiclass_classification_cifar10.pth"
cnn = torch.load(model_path)


transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((32,32))      # The same size as in  CIFAR10 dataset, as used in the training set
])


# We load the images for prediction/inference from local disk
img_1_maia = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\My Images\ForDeplymentCIFAR10\maia2023.jpg").type(torch.FloatTensor).unsqueeze(0)
img_2_bird = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\My Images\ForDeplymentCIFAR10\IMG_20190720_100620.jpg").type(torch.FloatTensor).unsqueeze(0)
img_3_bird = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\My Images\ForDeplymentCIFAR10\IMG_20190720_100643.jpg").type(torch.FloatTensor).unsqueeze(0)
img_4_cat = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\My Images\ForDeplymentCIFAR10\IMG_20190911_133815.jpg").type(torch.FloatTensor).unsqueeze(0)
img_5_car = torchvision.io.read_image("C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\My Images\ForDeplymentCIFAR10\IMG_20220304_134719.jpg").type(torch.FloatTensor).unsqueeze(0)


# Apply transformations to the images
img_1_maia = transform(img_1_maia)
img_2_bird = transform(img_2_bird)
img_3_bird = transform(img_3_bird)
img_4_cat = transform(img_4_cat)
img_5_car = transform(img_5_car)


# Make prediction for the images
prediction_1 = cnn(img_1_maia)
prediction_2 = cnn(img_2_bird)
prediction_3 = cnn(img_3_bird)
prediction_4 = cnn(img_4_cat)
prediction_5 = cnn(img_5_car)


# The results are:
print(f"\n\n\nPrediction is: {prediction_1} as probability and {prediction_1.argmax(1).item()} as class/label. It must be: DOG ")
print(f"\n\n\nPrediction is: {prediction_2} as probability and {prediction_2.argmax(1).item()} as class/label. It must be: BIRD.")
print(f"\n\n\nPrediction is: {prediction_3} as probability and {prediction_3.argmax(1).item()} as class/label. It must be: BIRD.")
print(f"\n\n\nPrediction is: {prediction_4} as probability and {prediction_4.argmax(1).item()} as class/label. It must be: CAT.")
print(f"\n\n\nPrediction is: {prediction_5} as probability and {prediction_5.argmax(1).item()} as class/label. It must be: CAR.")
