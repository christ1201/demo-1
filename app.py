from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("catboostModel.pkl")

@app.route('/')
def index():
    return send_from_directory('static', 'main.html')

@app.route('/predict_fare', methods=['POST'])
def predict_fare():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
    
    # Extract input features from JSON data
    weekday = data.get('weekday')
    hour = data.get('hour')
    minute = data.get('minute')
    distance = data.get('distance')  # in meters
    pickup_area = data.get('pickup_area')
    dropoff_area = data.get('dropoff_area')
    
    # Preprocess input data if needed
    # Example: Convert weekday to numeric value (0-6)
    weekdays = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    weekday_numeric = weekdays.get(weekday)
    
    # Create DataFrame with input features
    input_data = pd.DataFrame({
        'weekday': [weekday_numeric],
        'time': [hour * 60 + minute],  # Convert hour and minute to total minutes
        'distance': [distance],
        'pickup_area': [pickup_area],
        'dropoff_area': [dropoff_area]
    })
    
    # Make fare prediction
    predicted_fare = model.predict(input_data)[0]
    
    return jsonify({'predicted_fare': predicted_fare})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))