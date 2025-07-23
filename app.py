# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the crime rate prediction model and related data
try:
    # Load model
    crime_model_path = os.path.join('models', 'crime_rate_model.pkl')
    with open(crime_model_path, 'rb') as f:
        crime_model = pickle.load(f)
    
    # Load scaler
    crime_scaler_path = os.path.join('models', 'crime_scaler.pkl')
    with open(crime_scaler_path, 'rb') as f:
        crime_scaler = pickle.load(f)
    
    # Load feature names
    crime_features_path = os.path.join('models', 'crime_features.pkl')
    with open(crime_features_path, 'rb') as f:
        crime_features = pickle.load(f)
    
    print(f"Crime model loaded successfully with {len(crime_features)} features: {crime_features[:5]}...")
except Exception as e:
    print(f"Error loading crime model: {e}")
    crime_model = None
    crime_scaler = None
    crime_features = []

# Load the house price prediction model and related data
try:
    # Load model
    house_price_model_path = os.path.join('models', 'chennai_house_price_model.pkl')
    with open(house_price_model_path, 'rb') as f:
        house_price_model = pickle.load(f)
    
    # Load scaler
    house_price_scaler_path = os.path.join('models', 'house_price_scaler.pkl')
    with open(house_price_scaler_path, 'rb') as f:
        house_price_scaler = pickle.load(f)
    
    # Load feature names
    house_price_features_path = os.path.join('models', 'house_price_features.pkl')
    with open(house_price_features_path, 'rb') as f:
        house_price_features = pickle.load(f)
    
    print(f"House price model loaded successfully with {len(house_price_features)} features: {house_price_features[:5]}...")
except Exception as e:
    print(f"Error loading house price model: {e}")
    house_price_model = None
    house_price_scaler = None
    house_price_features = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crime_prediction')
def crime_prediction():
    # Pass the feature names to the template
    return render_template('crime_prediction.html', features=crime_features)

@app.route('/house_price_prediction')
def house_price_prediction():
    # Pass the feature names to the template
    return render_template('house_price_prediction.html', features=house_price_features)

@app.route('/predict_crime', methods=['POST'])
def predict_crime():
    if crime_model is None:
        return jsonify({'error': 'Crime model not loaded'})
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create a dictionary to hold feature values
        feature_values = {}
        
        # Handle each feature from the form
        for feature in crime_features:
            if feature in data:
                try:
                    feature_values[feature] = float(data[feature])
                except ValueError:
                    return jsonify({'error': f'Invalid input for {feature}'})
            else:
                # If feature not provided, use 0 as default
                feature_values[feature] = 0
        
        # Create a DataFrame with the correct feature order
        input_df = pd.DataFrame([feature_values])
        
        # Ensure the DataFrame has all the features needed (in the correct order)
        for feature in crime_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features used during training and in the same order
        input_df = input_df[crime_features]
        
        # Scale the input
        scaled_input = crime_scaler.transform(input_df)
        
        # Make prediction
        prediction = crime_model.predict(scaled_input)[0]
        
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    if house_price_model is None:
        return jsonify({'error': 'House price model not loaded'})
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create a dictionary to hold feature values
        feature_values = {}
        
        # Handle each feature from the form
        for feature in house_price_features:
            if feature in data:
                try:
                    feature_values[feature] = float(data[feature])
                except ValueError:
                    return jsonify({'error': f'Invalid input for {feature}'})
            else:
                # If feature not provided, use 0 as default
                feature_values[feature] = 0
        
        # Create a DataFrame with the correct feature order
        input_df = pd.DataFrame([feature_values])
        
        # Ensure the DataFrame has all the features needed (in the correct order)
        for feature in house_price_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features used during training and in the same order
        input_df = input_df[house_price_features]
        
        # Scale the input
        scaled_input = house_price_scaler.transform(input_df)
        
        # Make prediction
        prediction = house_price_model.predict(scaled_input)[0]
        
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/get_crime_features')
def get_crime_features():
    return jsonify({'features': crime_features})

@app.route('/get_house_price_features')
def get_house_price_features():
    return jsonify({'features': house_price_features})

if __name__ == '__main__':
    app.run(debug=True)