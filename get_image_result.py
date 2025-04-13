import os

from PIL import Image

from predict_model import PredictModel

model = PredictModel()

IMG_DIR = "datasets/rellay_smoke_original"

os.makedirs("results", exist_ok=True)
img_names = os.listdir(IMG_DIR)
for image_name in img_names:
    img = Image.open(os.path.join(IMG_DIR, image_name))
    r_image = model.get_image(img)
    r_image.save("results/" + image_name)
    print(" done!")
