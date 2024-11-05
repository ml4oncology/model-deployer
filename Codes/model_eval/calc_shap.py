"""
Script to generate and plot SHAP values
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to plot SHAP values for a single patient
def plot_shap_values_for_patient(patient_idx, shap_values, feature_names, result, fig_dir, top_n=10):
    # Extract SHAP values for the specific patient
    patient_shap_values = shap_values[patient_idx]
    
    # Sort the features by absolute SHAP values (for the patient)
    sorted_idx = np.argsort(np.abs(patient_shap_values))[::-1]
    
    # Select the top N most important features
    top_features = feature_names[sorted_idx][:top_n]
    top_shap_values = patient_shap_values[sorted_idx][:top_n]
    
    clinic_date=datetime.strftime(result['clinic_date'].iloc[patient_idx], '%d_%b_%y')
    
    # Create a bar plot for the top N most important features
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_shap_values, color="lightcoral")
    plt.xlabel('SHAP Value (Impact on model output)')
    plt.title(f'Patient clinic date: {clinic_date} - Top {top_n} Feature Importance based on SHAP values')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
    plt.show()
    
    plt_name = f"{fig_dir}/fig_{patient_idx}_{result['mrn'].iloc[patient_idx]}_{clinic_date}.png"
    plt.savefig(plt_name, bbox_inches='tight')
    plt.close()


def calc_plot_mean_shap_values(X_test, models, df_patient_info, fig_dir):
    
    X_test_numeric = X_test.applymap(lambda x: 1 if x == True else x)
    X_test_numeric = X_test_numeric.applymap(lambda x: 0 if x == False else x)
    
    shap_values_list = []
    for m in models:
        explainer = shap.Explainer(m.predict, X_test_numeric)
        shap_values = explainer.shap_values(X_test_numeric)
        shap_values_list.append(shap_values)
    
    shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)

    feature_names = X_test_numeric.columns
    
    # shap.summary_plot(shap_values, X_apply)
    
    shap_values_with_columns = pd.DataFrame(shap_values, columns = feature_names)
    vals = np.abs(shap_values_with_columns.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    # shap_importance.head()
    
    # Loop through each patient and plot their individual SHAP values
    for patient_idx in range(shap_values.shape[0]):
        plot_shap_values_for_patient(patient_idx, shap_values, feature_names, df_patient_info, fig_dir, top_n=10)
        
    return shap_values
    