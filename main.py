from binaryClassifierModel import PCABinaryClassifier
from BaseModel_model_Input import model_input
from BaseModel_predList import PredictionList
from fastapi import FastAPI, File, UploadFile
from typing import List
import json
import pickle
import torch
import pandas as pd
import numpy as np
import io

app = FastAPI()

SCALER_DIR="scalers/"
## Loading the scalars
scaler_file= f'{SCALER_DIR}scaler.pkl'
with open(scaler_file, 'rb') as f:
  scaler = pickle.load(f)

## Loading the frequency-map
freq_file = f'{SCALER_DIR}frequency_map.pkl'
with open(freq_file, 'rb') as f:
  frequency_map = pickle.load(f)

## Loading the PCA
pca_file= f'{SCALER_DIR}pca.pkl'
with open(pca_file, 'rb') as f:
  pca = pickle.load(f)

MODEL_DIR = "models/"
## Create model instance
model = PCABinaryClassifier()
## Loading the model.
model_2_file= f'{MODEL_DIR}model_2.pth'
state_dict = torch.load(model_2_file,weights_only=True)
model.load_state_dict(state_dict)


## Helper functions

# Function to apply frequency mapping
def apply_frequency_mapping(data: pd.DataFrame, frequency_map: dict, categorical_cols: list, drop_original: bool = True) -> pd.DataFrame:
    """
    Applies frequency mapping to the specified categorical columns in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame on which frequency mapping is applied.
    - frequency_map (dict): Dictionary containing frequency mappings for categorical columns.
    - categorical_cols (list): List of categorical columns to map.
    - drop_original (bool): If True, drops the original categorical columns after mapping. Default is True.

    Returns:
    - pd.DataFrame: Transformed DataFrame with frequency-mapped columns.
    """
    for col in categorical_cols:
        # Ensure the column exists in the data
        if col in data.columns:
            # Apply the frequency mapping using the values inside the column
            data[f"{col}_frequency"] = data[col].map(frequency_map).fillna(0)  # Default to 0 if not found
            if drop_original:
                data.drop(columns=col, inplace=True)   
    return data



@app.post('/predForOne')
def pred_for_one(input_parameters:model_input ):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    df = pd.DataFrame([input_dictionary])
    df = apply_frequency_mapping(df, frequency_map,  categorical_cols=["tcp_flags", "tcp_payload", "ip_src", "ip_dst", "mqtt_clientid", "mqtt_topic", "tcp_checksum"])
    X_pred=scaler.fit_transform(df)
    X_pred=pca.transform(X_pred)

    # Convert X_pred to a PyTorch tensor
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        pred_logits=model(X_pred_tensor).squeeze(dim=1)
        predictions=torch.round(torch.sigmoid(pred_logits))
    
    # Return the predictions as a list, as it is still a tensor
    return {"predictions": predictions.tolist()}

    
@app.post('/predForCSV')
async def pred_for_csv(file: UploadFile = File(...)):

    # Read the uploaded CSV file into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Apply frequency mapping to categorical columns
    df = apply_frequency_mapping(df, frequency_map, categorical_cols=["tcp_flags", "tcp_payload", "ip_src", "ip_dst", "mqtt_clientid", "mqtt_topic", "tcp_checksum"])
    
    # Scale the data
    X_pred = scaler.fit_transform(df)
    
    # Apply PCA transformation
    X_pred = pca.transform(X_pred)
    
    # Convert X_pred to a PyTorch tensor
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

    # Set the model to evaluation mode
    model.eval()
    
    # Perform predictions in a no_grad context (for inference)
    with torch.no_grad():
        pred_logits = model(X_pred_tensor).squeeze(dim=1)  # Remove unnecessary dimensions
        predictions = torch.round(torch.sigmoid(pred_logits))  # Apply sigmoid and round for binary classification

    # Convert predictions to a list and return as JSON
    return {"predictions": predictions.tolist()}


@app.post('/getNet')
async def get_net_outcomes(data:PredictionList):
    
    # Get the pred_list from json.
    pred_list = data.inp
    ## Get the count of attack and non-attack
    attack_count = pred_list.count(1)
    non_attack_count = pred_list.count(0)

    return {"attacks":attack_count,"non-attacks":non_attack_count}

@app.post('/getTrueCount')
async def pred_for_csv(file: UploadFile = File(...)):

    # Read the uploaded CSV file into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    true_val_list = df["label"].tolist()
    return {"true_values":true_val_list}
