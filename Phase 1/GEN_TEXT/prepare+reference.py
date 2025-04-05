import pandas as pd
import re

# File paths
ptbxl_file_path = "ptbxl_database.csv"
scp_statements_file_path = "scp_statements.csv"
existing_output_file_path = "/Users/sanjaibalajee/Development/GitHub/Multimodal_ECG/GEN_TEXT/ecg_reports_output.csv"

# Load the necessary CSV files
ptbxl_df = pd.read_csv(ptbxl_file_path)
scp_statements_df = pd.read_csv(scp_statements_file_path, index_col=0)
output_df = pd.read_csv(existing_output_file_path)

# Filter `scp_statements_df` to only include relevant diagnostic classes
scp_statements_df = scp_statements_df[scp_statements_df['diagnostic'] == 1]

def aggregate_diagnostic_class(y_dic):
    """Aggregates diagnostic classes based on scp_codes."""
    tmp = []
    for code in y_dic.keys():
        if code in scp_statements_df.index:
            tmp.append(scp_statements_df.loc[code, 'diagnostic_class'])
    return list(set(tmp))

def format_reference_report(row):
    """Generates a structured reference report using diagnostic superclass."""
    # Access the necessary fields in ptbxl_df using the ECG ID
    ecg_id = row['ecg_id']
    ptbxl_row = ptbxl_df[ptbxl_df['ecg_id'] == ecg_id].iloc[0]

    # Process diagnostic superclass from scp codes
    scp_codes = eval(ptbxl_row['scp_codes'])  # Convert string dict to dictionary
    diagnostic_classes = aggregate_diagnostic_class(scp_codes)
    diagnostic_text = ', '.join(diagnostic_classes) if diagnostic_classes else "Unknown diagnostic class"
    
    # Format other components of the report
    heart_axis = ptbxl_row['heart_axis'] if not pd.isnull(ptbxl_row['heart_axis']) else 'no heart axis data'
    infarction_status = "Infarction detected" if not pd.isnull(ptbxl_row['infarction_stadium1']) else "No infarction detected"

    # Construct the reference report
    reference_report = (
        f"| Diagnostic class: {diagnostic_text}; heart axis: {heart_axis}; infarction status: {infarction_status} |"
    )
    
    return reference_report

# Apply the function to add a reference report column to the output DataFrame
output_df['reference_report'] = output_df.apply(format_reference_report, axis=1)

# Save the updated DataFrame back to the output file
output_df.to_csv(existing_output_file_path, index=False)

print("Reference report column added using diagnostic classes from `scp_statements.csv`.")
