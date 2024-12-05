# Healthcare Network Attack Predictor

This project focuses on predicting and differentiating between normal and malicious network traffic within healthcare environments. The tool leverages machine learning models to classify network activities and provide actionable insights for healthcare security and privacy.

---

## 📂 Project Folder Structure

The project is organized as follows:

```
├── app.py  <!-- Main Streamlit file for the user interface -->
├── BaseModel_model_Input.py <!-- Defines Pydantic model for type-safe input -->
├── BaseModel_predList.py <!-- Intermediary Pydantic model for type-safe predictions -->
├── binaryClassifierModel.py <!-- Contains the binary classification model used in the project -->
├── main.py <!-- API file for FastAPI -->
├── models
│   └── model_2.pth <!-- Trained PyTorch model file -->
├── requirements.txt <!-- Lists project dependencies -->
└── scalers
    ├── frequency_map.pkl <!-- Frequency map for categorical features (used for encoding) -->
    ├── pca.pkl <!-- Trained PCA model for dimensionality reduction -->
    └── scaler.pkl <!-- Trained scaler for feature scaling -->
└── datasets
    ├── labelled_22000.csv
    └── unlabelled_22000.csv
```

# About the Dataset and Objective

## Dataset Overview

We use the [IoT Healthcare Security Dataset](https://www.kaggle.com/datasets/faisalmalik/iot-healthcare-security-dataset), which contains network traffic data from IoT medical devices. It includes:

- **Normal Traffic (0):** Routine operations, like accessing patient records or updating inventory.
- **Malicious Traffic (1):** Suspicious activities, such as unauthorized access or malware injection.

The dataset features TCP and MQTT protocol fields, timestamps, and frequency-based metrics, and about 50 features enabling detailed analysis and model training.

---

## Objective

Our goal is to analyze, preprocess, and visualize this dataset to identify patterns and anomalies. Using machine learning techniques, specifically Artificial Neural Networks (ANN), we aim to classify network traffic as either normal or malicious, contributing to enhanced cybersecurity for IoT medical devices.
Markdown

## Detailed Metrics

| Metric                 | Value    |
| ---------------------- | -------- |
| Precision (Non-Attack) | 0.9768   |
| Recall (Non-Attack)    | 0.7935   |
| F1-Score (Non-Attack)  | 0.8757   |
| Support (Non-Attack)   | 108568.0 |
| Precision (Attack)     | 0.7769   |
| Recall (Attack)        | 0.9745   |
| F1-Score (Attack)      | 0.8646   |
| Support (Attack)       | 80126.0  |

## Additional Metrics

| Metric              | Value  |
| ------------------- | ------ |
| Specificity         | 0.7935 |
| False Positive Rate | 0.2065 |
| False Negative Rate | 0.0255 |

# Additional Details

For more in-depth information, including the **Confusion Matrix** and **AUC-ROC** analysis, please refer to the PDF provided in the project directory.

# How to Use the Project

Follow these steps to set up and run the project:

1. **Set Up the Environment**  
   Ensure your device has Conda installed for easy environment management (Python environments work too).

   ```bash
   conda create --name health_sec_env python=3.10 -y
   conda activate health_sec_env

   ```

2. **Install Dependencies**
   Install the required packages using requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Server**

   ```bash
   uvicorn main:app --reload
   ```

4. **Run the Streamlit UI**

   ```bash
   streamlit run app.py
   ```

5. **Datasets Provided**

   In the Datasets folder, you will find:

   - A labelled dataset with 22,000 datapoints for training and evaluation.
   - An unlabelled dataset for inference and performance testing.

6. **Perform Inference**

   In the running Streamlit app:

   - Upload the CSV file in the designated upload section.
   - Run the inference to get predictions.


# Conclusion  

- The model effectively detects TCP-based attacks in healthcare networks, achieving a high AUC score of >=**0.90**, indicating strong performance in distinguishing between normal and malicious traffic.  
- With a sensitivity of **0.97**, it correctly identifies **97%** of actual attacks, minimizing false negatives.  
- Data preprocessing using PCA enhanced the model by reducing dimensionality and improving efficiency.  
- Data visualization provided key insights into patterns, aiding feature selection and model development.  

This tool shows great potential for strengthening healthcare security against cyberattacks.
