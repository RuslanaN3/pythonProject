import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from model_instance import  get_model_instance
import torch

app = Flask(__name__)

# Load
PATH_TO_MODEL = "state_dict_model.pt"
model = get_model_instance(3)
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.eval()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            pred = render_prediction(prediction_idx)
            #return jsonify({'class_id': class_id, 'class_name': class_name})

def transform_image(infile):
    input_transforms = [transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction


def render_prediction(prediction_idx):
    # transform prediction
    # stridx = str(prediction_idx)
    # class_name = 'Unknown'
    # if img_class_map is not None:
    #     if stridx in img_class_map is not None:
    #         class_name = img_class_map[stridx][1]
    #
    # return prediction_idx, class_name
