from flask import Flask, request, jsonify
from PIL import Image
from api.pred.functions import transform_image, get_prediction
from api.pred.model_instance import get_model_instance
import torch

app = Flask(__name__)

device = torch.device('cpu')
# Load
PATH_TO_MODEL = "api/pred/state_dict_model.pt"
model = get_model_instance(3)
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
model.eval()


@app.route('/api/test')
def hello_world():
    print(request.data)
    return 'Hello, World!'


@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            img = Image.open(file.stream)
            pred_boxes, pred_labels, pred_th_scores = get_prediction(img, 0.6, model)
            return jsonify({'msg': 'success',
                            'size': [img.width, img.height],
                            'pred_boxes': str(pred_boxes),
                            'pred_labels': str(pred_labels),
                            "pred_th_scores": str(pred_th_scores)})
