"""
Save and display model performance results
"""

import pandas as pd
# from tabulate import tabulate

def save_and_display_model_results(model_results, perf_filename):
    """
    Save and display model performance results.

    Parameters:
    model_results (list of dict): A list where each dict contains model name and its performance metrics.
    filename (str): The filename for saving the CSV file.

    Example:
    model_results = [
        {"Model": "Model A", "Accuracy": 0.95, "Precision": 0.92, "Recall": 0.93},
        {"Model": "Model B", "Accuracy": 0.90, "Precision": 0.88, "Recall": 0.85}
    ]
    save_and_display_model_results(model_results)
    """
    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(model_results)

    # Save the DataFrame to a CSV file
    results_df.index.rename('idx', True)
    results_df.to_csv(perf_filename)
    print(f"Performance metrics saved to {perf_filename}.")

    # # Display the DataFrame in tabular format
    # print(tabulate(results_df, headers='keys', tablefmt='grid'))
