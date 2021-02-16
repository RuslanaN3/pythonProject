from flask import Flask, jsonify, request

from api.pred.functions import *
from model_instance import get_model_instance
import torch


app = Flask(__name__)

# Load
PATH_TO_MODEL = "api/pred/state_dict_model.pt"
model = get_model_instance(3)
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.eval()


@app.route('/api/test')
def hello_world():
    return 'Hello, World!'


@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            pred = render_prediction(prediction_idx)
            # return jsonify({'class_id': class_id, 'class_name': class_name})
