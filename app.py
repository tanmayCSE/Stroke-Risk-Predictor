# # -*- coding: utf-8 -*-
# """
# Created on [Date]
# @author: [Your Name]
# """

# # 1. Library imports
# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import tensorflow as tf
# import pandas as pd
# import numpy as np

# # 2. Create the app object
# app = FastAPI()

# # Load the preprocessing objects (scaler and oversampler)
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Load the trained TensorFlow model
# model = tf.keras.models.load_model('stroke_prediction_model.keras')

# # 3. Create Pydantic model for incoming JSON request data
# class StrokePredictionInput(BaseModel):
#     age: float
#     hypertension: int
#     heart_disease: int
#     avg_glucose_level: float
#     bmi: float
#     gender_enc: int
#     No: int
#     Yes: int
#     Govt_job: int
#     Never_worked: int
#     Private: int
#     Self_employed: int
#     children: int
#     Rural: int
#     Urban: int
#     Unknown: int
#     formerly_smoked: int
#     never_smoked: int
#     smokes: int

# # 4. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Welcome to the Stroke Prediction API'}

# # 5. Route with a single parameter, returns the parameter within a message
# #    Located at: http://127.0.0.1:8000/{name}
# @app.get('/{name}')
# def get_name(name: str):
#     return {'message': f'Welcome, {name}!'}

# # 6. Expose the prediction functionality, make a prediction from the passed
# #    JSON data and return the predicted risk (stroke or no stroke)
# @app.post('/predict')
# def predict_stroke(data: StrokePredictionInput):
#     data_dict = data.dict()
#     input_data = pd.DataFrame([data_dict])

#     # Ensure the input matches the exact feature set the model was trained on
#     input_data = input_data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_enc', 
#                              'No', 'Yes', 'Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children', 
#                              'Rural', 'Urban', 'Unknown', 'formerly_smoked', 'never_smoked', 'smokes']]

#     # List of features that need to be scaled based on the model's training
#     scale_features = ['bmi', 'avg_glucose_level', 'age']

#     # Reorder input_data columns to match the scaler's expected order for scaling features
#     input_data = input_data[['bmi', 'avg_glucose_level', 'age'] + [col for col in input_data.columns if col not in scale_features]]

#     # Preprocess the input (scaling)
#     input_data[scale_features] = scaler.transform(input_data[scale_features])

#     # Get prediction
#     sample_prediction = (model.predict(input_data) > 0.5).astype("int32")

#     # Return the prediction result
#     if sample_prediction[0][0] == 1:
#         prediction = "The person is at risk of having a stroke."
#     else:
#         prediction = "The person is not at risk of having a stroke."

#     return {'prediction': prediction}

# # 7. Run the API with uvicorn
# #    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
    
# # To run, use the command:
# # uvicorn app:app --reload




# -*- coding: utf-8 -*-
"""
Created on [Date]
@author: [Your Name]
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging

# 2. Create the app object
app = FastAPI()

# Define file paths for model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'stroke_prediction_model.keras')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Load the preprocessing objects (scaler and oversampler)
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load the trained TensorFlow model
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    raise

# 3. Create Pydantic model for incoming JSON request data
class StrokePredictionInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender_enc: int
    No: int
    Yes: int
    Govt_job: int
    Never_worked: int
    Private: int
    Self_employed: int
    children: int
    Rural: int
    Urban: int
    Unknown: int
    formerly_smoked: int
    never_smoked: int
    smokes: int

# 4. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome to the Stroke Prediction API'}

# 5. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/{name}
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Welcome, {name}!'}

# 6. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted risk (stroke or no stroke)
@app.post('/predict')
def predict_stroke(data: StrokePredictionInput):
    try:
        data_dict = data.dict()
        input_data = pd.DataFrame([data_dict])

        # Ensure the input matches the exact feature set the model was trained on
        input_data = input_data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_enc', 
                                 'No', 'Yes', 'Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children', 
                                 'Rural', 'Urban', 'Unknown', 'formerly_smoked', 'never_smoked', 'smokes']]

        # List of features that need to be scaled based on the model's training
        scale_features = ['bmi', 'avg_glucose_level', 'age']

        # Reorder input_data columns to match the scaler's expected order for scaling features
        input_data = input_data[['bmi', 'avg_glucose_level', 'age'] + [col for col in input_data.columns if col not in scale_features]]

        # Preprocess the input (scaling)
        input_data[scale_features] = scaler.transform(input_data[scale_features])

        # Get prediction
        sample_prediction = (model.predict(input_data) > 0.5).astype("int32")

        # Return the prediction result
        if sample_prediction[0][0] == 1:
            prediction = "The person is at risk of having a stroke."
        else:
            prediction = "The person is not at risk of having a stroke."

        return {'prediction': prediction}
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return {'error': 'Internal server error, please try again later.'}, 500

# 7. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# To run, use the command:
# uvicorn app:app --reload