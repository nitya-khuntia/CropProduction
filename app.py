from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename='app.log', 
                    filemode='w')

app = Flask(__name__)

# Load the trained model and encoder
MODEL_PATH = 'trained_model.pkl'
ENCODER_PATH = 'encoder.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    trained_model = pickle.load(model_file)

with open(ENCODER_PATH, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Load the column names from the serialized file
with open('columns.pkl', 'rb') as columns_file:
    trained_columns = pickle.load(columns_file)

# Preprocessing function
def preprocess_data(data):
    try:
        # Drop the 'District_Name' and 'Production' columns if they exist
        data = data.drop(columns=['District_Name', 'Production'], errors='ignore')
        
        # Check if 'Production' column exists before dropping rows based on it
        if 'Production' in data.columns:
            data = data.dropna(subset=['Production']).reset_index(drop=True)
        
        # Apply logarithmic transformation to 'Area'
        if 'Area' in data.columns:
            data['Area'] = data['Area'].apply(lambda x: np.log(x + 1))

        
        # Apply one-hot encoding to categorical columns if they exist
        categorical_columns = ['State_Name', 'Season', 'Crop']
        if set(categorical_columns).issubset(data.columns):
            encoded_features = encoder.transform(data[categorical_columns])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
            
            # Concatenate the original dataframe with the encoded dataframe and drop original categorical columns
            data = pd.concat([data, encoded_df], axis=1)
            data.drop(columns=categorical_columns, inplace=True)

        # Ensure all columns are present and in the correct order
        for col in trained_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[trained_columns]

        return data
        
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

@app.route('/')
def index():
    return open('crop_prediction_frontend.html').read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)

        print("Received data:", data)

        
        # Convert data to DataFrame format for preprocessing
        sample_data = pd.DataFrame([data])
        
        # Preprocess the sample data
        sample_data_encoded = preprocess_data(sample_data)
        print("Preprocessed data:", sample_data_encoded)

    

        # Check if preprocessing was successful
        if sample_data_encoded is None:
            raise ValueError("Error during data preprocessing.")
        
        # Ensure sample data has the same columns as the training data
        sample_data_encoded = sample_data_encoded.reindex(columns=trained_columns, fill_value=0)
        
        # Predict production using the loaded model
        predicted_production = trained_model.predict(sample_data_encoded)
        
        # Convert the logarithmic-transformed prediction back to its original scale
        predicted_production = np.exp(predicted_production) - 1
        
        # Return the predicted production
        return jsonify({'predicted_production': predicted_production[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
