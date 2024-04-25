from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from datetime import datetime
import joblib
import os
import catboost
import webbrowser
import functions_framework
import json

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("catboostModel.pkl")


@app.route('/home', methods=['GET'])
def index():
    # filename = 'index.html'
    # webbrowser.open_new_tab(filename)
    return send_from_directory('static', 'index.html')

@app.route('/predict_fare', methods=['POST'])
# @app.route('/predict', methods=['POST'])
def predict_fare():
    try:
        predict = request.get_json(silent=True)
        data = predict
    except Exception as e:
        return jsonify({'error': 'Invalid JSON data'}), 400

    print(f"Received data: Weekday: {data['weekday']}, Hour: {data['hour']}, Minute: {data['minute']}, Distance: {data['distance']}, Pickup_area: {data['pickup_area']}, dropoff_area: {data['dropoff_area']}")
    print(f"Data received successfully!!")

    # if not data:
    #     return jsonify({'error': 'No input data provided'}), 400
    
    # Extract input features from JSON data
    jsonify(data)

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
    # json.dumps(input_data)
    jsonify(input_data)
    try:
        predicted_fare = model.predict(input_data)[0]
        # predicted_fare = model.predict(json.dumps(input_data))
    except Exception as e:
        return jsonify({'error': "There an error!"}), 400

    return jsonify({'predicted_fare': predicted_fare})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))