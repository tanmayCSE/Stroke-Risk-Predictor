from pydantic import BaseModel
import pickle
import tensorflow as tf
import pandas as pd

# 2. Class which describes Stroke prediction input features
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