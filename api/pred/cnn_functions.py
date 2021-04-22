import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch

from api.pred.cnn_model import get_model


def load_patches_coord(fileName):
    patches_coord_file = pd.read_csv(fileName, delimiter=";")
    df = pd.DataFrame(patches_coord_file)
    patches_coord = df.to_dict("records")
    return patches_coord


def get_img_patches(image, patches_coord):
    patches = []
    for pc in patches_coord:
        roi = image[pc["Y"]:pc["Y"] + pc["H"], pc["X"]:pc["X"] + pc["W"]]
        patches.append(roi)
    return patches


def get_predictions(img_patches):
    predictions = []
    for ip in img_patches:
        ip = Image.fromarray(ip)
        trans = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Resize((28, 28))])
        im = trans(ip)
        im = im.unsqueeze(1)
        model = get_model()
        pred = model(im)
        _, pr = torch.max(pred.data, 1)
        predictions.append(pr.item())
    return predictions


##prediction
def get_slot_states(file):
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    patch_coord = load_patches_coord("api/resources/camera5s.csv")
    img_patches = get_img_patches(img, patch_coord)
    preds = get_predictions(img_patches)
    return preds
