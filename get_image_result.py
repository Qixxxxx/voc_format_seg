import os

from PIL import Image

from predict_model import PredictModel

model = PredictModel()

os.makedirs("results", exist_ok=True)
img_names = os.listdir("datasets/viewed_images")
for image_name in img_names:
    img = Image.open(os.path.join("datasets/viewed_images/", image_name))
    r_image = model.get_image(img)
    r_image.save("results/" + image_name)
    print(" done!")
