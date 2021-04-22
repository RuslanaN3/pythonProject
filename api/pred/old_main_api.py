from flask import Flask, request, jsonify
from PIL import Image
from api.pred.cnn_functions import get_slot_states
from api.pred.convert_data import convert_data
from api.pred.functions import get_prediction
from api.pred.model_instance import get_model_instance
import torch

app = Flask(__name__)


# device = torch.device('cpu')
# Load
# PATH_TO_MODEL = "api/pred/state_dict_model.pt"
# model = get_model_instance(3)
# model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
# model.eval()

#
# @app.route('/api/test')
# def hello_world():
#     print(request.data)
#     return 'Hello, World!'


# @app.route('/api/predictn', methods=['POST'])
# def predict_rcnn():
#     if request.method == 'POST':
#         print(request.data)
#         print(request.form)
#         print("file", request.files['file'])
#         file = request.files['file']
#         print(file)
#         if file is not None:
#             img = Image.open(file.stream)
#             pred_boxes, pred_labels, pred_th_scores = get_prediction(img, 0.6, model)
#             print(pred_boxes)
#             return jsonify({'msg': 'success',
#                             'size': [img.width, img.height],
#                             'pred_boxes': str(pred_boxes),
#                             'pred_labels': str(pred_labels),
#                             "pred_th_scores": str(pred_th_scores)})


@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            preds = []
            preds = get_slot_states(file)
            parking_slots_states = convert_data(preds)
            return jsonify(parking_slots_states)
