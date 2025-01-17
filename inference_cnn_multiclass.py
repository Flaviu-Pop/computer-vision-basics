import torch
import torchvision

from torchvision import transforms
from cnn_multiclass_classification import CnnMultiClassClassification

torch.set_printoptions(precision=5, sci_mode=False)

# We load the model
model_path = "../computer-vision-basics/cnn_multiclass_classification_cifar10.pth"
cnn = torch.load(model_path)

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((32, 32))  # The same size as in  CIFAR10 dataset, as used in the training set
])


def make_prediction(model, image):
    image = image.type(torch.FloatTensor).unsqueeze(0)
    image = transform(image)
    prediction = model(image)

    print(f"\n\n\nFirst: the probability distribution is: {prediction}. "
          f"\nSecond: the class/label predicted is: {prediction.argmax(1).item()}.")


# We load the images for prediction/inference from local disk
image_root_path = "../computer-vision-basics/single_prediction/"

img_1_maia = torchvision.io.read_image(image_root_path + "maia2023.jpg")
img_2_bird = torchvision.io.read_image(image_root_path + "IMG_20190720_100620.jpg")
img_3_bird = torchvision.io.read_image(image_root_path + "IMG_20190720_100643.jpg")
img_4_cat = torchvision.io.read_image(image_root_path + "IMG_20190911_133815.jpg")
img_5_car = torchvision.io.read_image(image_root_path + "IMG_20220304_134719.jpg")


make_prediction(cnn, img_1_maia)
print("It must be a DOG")

make_prediction(cnn, img_2_bird)
print("It must be a BIRD")

make_prediction(cnn, img_3_bird)
print("It must be a BIRD")

make_prediction(cnn, img_4_cat)
print("It must be a CAT")

make_prediction(cnn, img_5_car)
print("It must be a CAR")
