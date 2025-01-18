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

    print(f"\n\n\nFirst: the probability distribution is: \n{prediction}. "
          f"\nSecond: the class/label predicted is: {prediction.argmax(1).item()}.")

    pred = "PLANE"
    if prediction.argmax(1).item() == 1:
        pred = "CAR"
    elif prediction.argmax(1).item() == 2:
        pred = "BIRD"
    elif prediction.argmax(1).item() == 3:
        pred = "CAT"
    elif prediction.argmax(1).item() == 4:
        pred = "DEER"
    elif prediction.argmax(1).item() == 5:
        pred = "DOG"
    elif prediction.argmax(1).item() == 6:
        pred = "FROG"
    elif prediction.argmax(1).item() == 7:
        pred = "HORSE"
    elif prediction.argmax(1).item() == 8:
        pred = "SHIP"
    elif prediction.argmax(1).item() == 9:
        pred = "TRUCK"

    return pred


# We load the images for prediction/inference from local disk
image_root_path = "../computer-vision-basics/single_prediction/"

image_1_dog = torchvision.io.read_image(image_root_path + "image_00.jpg")
image_2_bird = torchvision.io.read_image(image_root_path + "image_03.jpg")
image_3_bird = torchvision.io.read_image(image_root_path + "image_04.jpg")
image_4_cat = torchvision.io.read_image(image_root_path + "image_05.jpg")
image_5_car = torchvision.io.read_image(image_root_path + "image_08.jpg")

prediction = make_prediction(cnn, image_1_dog)
print("The prediction is " + prediction +" and it must be a DOG")

prediction = make_prediction(cnn, image_2_bird)
print("The prediction is " + prediction +" and it must be BIRD")

prediction = make_prediction(cnn, image_3_bird)
print("The prediction is " + prediction +" and it must be a BIRD")

prediction = make_prediction(cnn, image_4_cat)
print("The prediction is " + prediction +" and it must be a CAT")

prediction = make_prediction(cnn, image_5_car)
print("The prediction is " + prediction +" and it must be a CAR")
