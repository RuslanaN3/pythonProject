from flask import Flask, request, jsonify
from api.pred.cnn_functions import get_slot_states
from api.pred.convert_data import convert_data

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            preds = []
            preds = get_slot_states(file)
            parking_slots_states = convert_data(preds)
            return jsonify(parking_slots_states)
