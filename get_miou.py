import os

from PIL import Image
from tqdm import tqdm

from predict_model import PredictModel
from utils.metrics import compute_mIoU, show_results


if __name__ == "__main__":

    num_classes = 21
    classes_name_list = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                         "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                         "train", "tvmonitor"]

    dataset_path = 'datasets/VOCdevkit'
    miou_out_path = "miou_out"
    pred_dir = "miou_pr_dir"

    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    model = PredictModel()

    image_ids = open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(dataset_path, "VOC2007/SegmentationClass/")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        image = model.get_predict_score_image(image)
        image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")

    print("Get miou.")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, classes_name_list)
    print("Get miou done.")

    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, classes_name_list)
