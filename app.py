import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/predForCSV"  # Update with your FastAPI URL
NET_COUNT_URL ="http://127.0.0.1:8000/getNet" 
TRUE_COUNT_URL="http://127.0.0.1:8000/getTrueCount" 


# Initialize session state for persistence 
## Persistance of data related values
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "true_labels" not in st.session_state:
    st.session_state["ture_labels"] = None
if "attack_count" not in st.session_state:
    st.session_state["attack_count"] = None
if "non_attack_count" not in st.session_state:
    st.session_state["non_attack_count"] = None
if "true_attack_count" not in st.session_state:
    st.session_state["true_attack_count"] = None
if "true_non_attack_count" not in st.session_state:
    st.session_state["true_non_attack_count"] = None

## Persistance of web related values
if "pred_btn_pressed" not in st.session_state:
    st.session_state["pred_btn_pressed"] = None
if "truelab_btn_pressed" not in st.session_state:
    st.session_state["truelab_btn_pressed"] = None
if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame(columns=["Category", "Predictions", "True Labels"])
if "comparision_plot" not in st.session_state:
    st.session_state["comparision_plot"] = None
if "auc_plot" not in st.session_state:
    st.session_state["auc_plot"] = None
    


# Title
st.title("Network Traffic Attack Predictor")

st.subheader("Upload the CSV files")
# Upload CSV file
uploaded_true_file = st.file_uploader("Upload the 'file with true labels' CSV file", type=["csv"])
uploaded_pred_file = st.file_uploader("Upload the 'to be predicted upon' CSV file", type=["csv"])


if uploaded_pred_file is not None:
    st.write("Uploaded File:")
    st.write(uploaded_pred_file.name)  # Display file name

    # Read and show the uploaded CSV content
    df = pd.read_csv(uploaded_pred_file)
    st.write("File Content:")
    st.write(df)


if uploaded_true_file is not None and uploaded_pred_file is not None :
    st.subheader("Click the buttons run the predictive model and extract the true labels")
    # Display results in side-by-side columns
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Predictions"):
            try:
                # Prepare file for POST request
                files = {"file": (uploaded_pred_file.name, uploaded_pred_file.getvalue(), "text/csv")}

                # Make POST request to FastAPI
                response = requests.post(FASTAPI_URL, files=files)

                # Parse the response
                if response.status_code == 200:
                    response_data = response.json()
                    predictions = response_data.get("predictions", [])

                    # Store predictions in session state
                    st.session_state["predictions"] = predictions

                    count_response = requests.post(NET_COUNT_URL, json={"inp": predictions})

                    if count_response.status_code==200:
                        count_response= count_response.json()

                        # Store counts in session state
                        st.session_state["attack_count"] = count_response.get("attacks")
                        st.session_state["non_attack_count"] = count_response.get("non-attacks")

                        # Count "Attack" and "Non-Attack" predictions
                        attack_count = count_response.get("attacks")
                        non_attack_count = count_response.get("non-attacks")

                        st.session_state["pred_btn_pressed"] = True
                        # 
                    else:
                        st.error(f"Error: {count_response.status_code} - {count_response.text}")
                        
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    with col2:
        if st.button("Get True Label Count"):
            try:
                # Prepare file for POST request
                files = {"file": (uploaded_true_file.name, uploaded_true_file.getvalue(), "text/csv")}

                # Make POST request to FastAPI
                response = requests.post(TRUE_COUNT_URL, files=files)

                # Parse the response
                if response.status_code == 200:
                    response_data = response.json()
                    # print(true_count_list)
                    true_count_list = response_data.get("true_values", [])

                    # Store true_labels in session state
                    st.session_state["true_labels"] = true_count_list

                    true_counts = requests.post(NET_COUNT_URL, json={"inp": true_count_list})

                    if true_counts.status_code==200:
                        true_counts_data= true_counts.json()
                        # print(true_counts_data)

                        # # Store counts in session state
                        st.session_state["true_attack_count"] = true_counts_data.get("attacks")
                        st.session_state["true_non_attack_count"] = true_counts_data.get("non-attacks")

                        # # Count "Attack" and "Non-Attack" predictions
                        true_attack_count = true_counts_data.get("attacks")
                        true_non_attack_count = true_counts_data.get("non-attacks")

                        st.session_state["truelab_btn_pressed"]=True
                        # st.success("True Labels extracted!")
                        

                    else:
                        st.error(f"Error: {true_count.status_code} - {true_count.text}")
                        
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    if(st.session_state["pred_btn_pressed"]):
        st.success("Predictions Done!")
    if(st.session_state["truelab_btn_pressed"]):
        st.success("True Labels extracted!")
    




if st.session_state["attack_count"] is not None and st.session_state["true_attack_count"] is not None:
    st.subheader("Click the button to get a Comparision  ")
    if st.button("Show Predictions and True Labels"):

        # Create a DataFrame to display both prediction and true label results
        results_df = pd.DataFrame({
            "Category": ["Attacks", "Non-Attacks"],
            "Predictions": [st.session_state["attack_count"], st.session_state["non_attack_count"]],
            "True Labels": [st.session_state["true_attack_count"], st.session_state["true_non_attack_count"]]
        })  
        st.session_state["results_df"]=results_df

    if not st.session_state["results_df"].empty:
         st.table(st.session_state["results_df"])


# Function to generate and store the plot
def generate_comparision_plot():
    # Create the DataFrame
    graph_df = pd.DataFrame({
        "Category": ["True Attack", "Predicted Attack", "True Non-Attack", "Predicted Non-Attack"],
        "Count": [
            st.session_state["true_attack_count"],
            st.session_state["attack_count"],
            st.session_state["true_non_attack_count"],
            st.session_state["non_attack_count"]
        ]
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define colors for True and Predicted
    true_color = '#7986CB'  # Blue-grey
    predicted_color = '#566268'  # Darker grey

    # Plot the bars
    ax.barh(graph_df['Category'], graph_df['Count'], color=[true_color, predicted_color, true_color, predicted_color])

    # Customize the plot
    ax.set_xlabel('Count')
    ax.set_title('Comparison of Predicted vs True Labels')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Store the figure in session state
    st.session_state["comparision_plot"] = fig


# Display the graph only if counts are available in session state
if st.session_state["attack_count"] is not None and st.session_state["true_attack_count"] is not None:
    if st.button("Get Comparative Graph"):
        generate_comparision_plot()

    if st.session_state["comparision_plot"]:
        st.pyplot(st.session_state["comparision_plot"])




def generate_auc_plot():
        ## Get the ROC AUC curve
        fpr, tpr, thresholds = roc_curve(np.array(st.session_state["true_labels"]), np.array(st.session_state["predictions"]))
        roc_auc = auc(fpr, tpr)

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)

        st.session_state["auc_plot"] = plt


if st.session_state["attack_count"] is not None and st.session_state["true_attack_count"] is not None:
    if st.button("Get ROC Curve"):
        generate_auc_plot()

    if st.session_state["auc_plot"]:
        st.pyplot(st.session_state["auc_plot"])