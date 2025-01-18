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
    pred = "CAT"
    image = image.type(torch.FloatTensor).unsqueeze(0)
    image = transform(image)
    prediction = model(image)

    print(f"\nThe prediction is: {prediction.item()}.")
    if prediction.item() <= 0.5:
        pred = "DOG"

    return pred


# We load the images for prediction/inference from local disk
image_root_path = "../computer-vision-basics/single_prediction/"

image_1_dog = torchvision.io.read_image(image_root_path + "image_00.jpg")
image_2_dog = torchvision.io.read_image(image_root_path + "image_01.jpg")
image_3_cat = torchvision.io.read_image(image_root_path + "image_02.jpg")


prediction = make_prediction(cnn, image_1_dog)
print("The prediction is " + prediction + " and it must be a DOG.")

prediction = make_prediction(cnn, image_2_dog)
print("The prediction is " + prediction + " and it must be a DOG.")

prediction = make_prediction(cnn, image_3_cat)
print("The prediction is " + prediction + " and it must be a CAT.")
