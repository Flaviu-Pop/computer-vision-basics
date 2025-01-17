import torch
import torchvision

from torchvision import transforms
from cnn_binary_classification import CnnBinaryClassification

torch.set_printoptions(precision=5, sci_mode=False)

# We load the model
model_path = "../computer-vision-basics/cnn_binary_classification_catdog.pth"
cnn = torch.load(model_path)

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((64, 64))      # The same size as in  training
])


def make_prediction(model, image):
    image = image.type(torch.FloatTensor).unsqueeze(0)
    image = transform(image)
    prediction = model(image)

    print(f"\nThe prediction is: {prediction.item()}.")


# We load the images for prediction/inference from local disk
image_root_path = "../computer-vision-basics/single_prediction/"

img_1_maia = torchvision.io.read_image(image_root_path + "maia2023.jpg")
img_2_dog = torchvision.io.read_image(image_root_path + "cat_or_dog_1.jpg")
img_3_cat = torchvision.io.read_image(image_root_path + "cat_or_dog_2.jpg")


make_prediction(cnn, img_1_maia)
print("It must be DOG.")

make_prediction(cnn, img_2_dog)
print("It must be DOG.")

make_prediction(cnn, img_3_cat)
print("It must be CAT.")
