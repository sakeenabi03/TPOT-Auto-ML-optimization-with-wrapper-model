from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load models and scaler
with open('tpot_model.pkl', 'rb') as f:
    tpot_model = pickle.load(f)

with open('smaller_model.pkl', 'rb') as f:
    smaller_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def wrapper_model(input_features, tpot_model, smaller_model, scaler):
    # Check for missing features
    if np.isnan(input_features[1]):
        # Predict missing feature using smaller model
        input_features_for_small_model = input_features[[0, 2, 3, 4, 5, 6, 7]]
        input_features[1] = smaller_model.predict(scaler.transform(input_features_for_small_model.reshape(1, -1)))[0]
    
    # Pass complete features to TPOT model
    return tpot_model.predict(input_features.reshape(1, -1))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array([
        data.get('MedInc'), 
        data.get('HouseAge'), 
        data.get('AveRooms'), 
        data.get('AveBedrms'), 
        data.get('Population'), 
        data.get('AveOccup'), 
        data.get('Latitude'), 
        data.get('Longitude')
    ], dtype=np.float64)
    
    # Replace None with NaN for processing
    input_features = np.where(input_features == None, np.nan, input_features)
    
    # Make prediction
    predicted_price = wrapper_model(input_features, tpot_model, smaller_model, scaler)
    
    return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)
