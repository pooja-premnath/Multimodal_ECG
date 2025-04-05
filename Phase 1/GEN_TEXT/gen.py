import pandas as pd
import ast
import logging
from langchain_community.llms import Ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# llm = Ollama(model="llama3.1")
llm = Ollama(model="llama3.1")


df = pd.read_csv("ptbxl_database.csv")


def get_model_response(user_prompt, system_prompt, temperature=0.2):  
    prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        { system_prompt }
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        { user_prompt }
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Please ensure that all responses follow this structure:
        - Provide a brief introduction, outlining the main topic.
        - Present key details in a structured, bullet-point format if applicable.
        - End with a concise conclusion or next steps, depending on the query.
        <|eot_id|>
    """
    response = llm.invoke(prompt, temperature=temperature)
    return response


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# def generate_prompt(row):
#     diagnostic_superclass_mapping = {
#         '[NORM]': 'Normal ECG',
#         '[MI]': 'Myocardial Infarction',
#         '[STTC]': 'ST/T Change',
#         '[CD]': 'Conduction Disturbance',
#         '[HYP]': 'Hypertrophy'
#     }
#     report = row['report']
#     heart_axis = row['heart_axis'] if not pd.isnull(row['heart_axis']) else 'no heart axis data'
#     infarction = "No infarction detected" if pd.isnull(row['infarction_stadium1']) else "Infarction detected"
#     diagnostic_superclass_raw = row['diagnostic_superclass']
    

#     prompt = (f"Generate a concise 1-line ECG report for a patient. "
#               f"The report mentions: '{report}' Translate that to english first, heart axis as {heart_axis}, "
#               f"diagnostic superclass: {diagnostic_superclass_raw}, and {infarction}. Generate only one line report, nothing else. ")
    
#     return prompt
def generate_prompt(row):
    """Generates a refined prompt for consistent report generation with added relevant details."""
    
    # Extract available information for the prompt
    report = row['report']
    heart_axis = row['heart_axis'] if not pd.isnull(row['heart_axis']) else 'no heart axis data'
    infarction = "No infarction detected" if pd.isnull(row['infarction_stadium1']) else "Infarction detected"
    diagnostic_superclass_raw = row['diagnostic_superclass']
    
    # Optional additional information for detail
    rate = row.get('rate', 'unknown rate')  # Assuming rate information is available; replace with relevant field name
    rhythm_type = row.get('rhythm_type', 'rhythm type not specified')  # Assuming rhythm_type is present in dataset
    
    # Construct the prompt with placeholders for a structured format
    prompt = (
        f"Generate a concise 1-line ECG diagnostic report. "
        f"Use the following format closely: | Diagnostic class: [diagnostic class], heart axis: [heart axis value], "
        f"infarction status: [infarction status], rate: [rate value], rhythm: [rhythm type] |. "
        f"Translate and format as needed:\n"
        f"- Diagnostic class based on '{report}': {diagnostic_superclass_raw}.\n"
        f"- Heart axis value: {heart_axis}.\n"
        f"- Infarction status: {infarction}.\n"
        f"- Rate: {rate}.\n"
        f"- Rhythm: {rhythm_type}.\n"
        f"Return the report strictly in one line and match the example format as closely as possible."
    )
    
    return prompt




# def generate_report(row):
#     user_prompt = generate_prompt(row)
#     system_prompt = "Strictly 1 line only. Generate only report, nothing else. Start main report with | and end with |. Do not include any other information. Do not use the phrase 'ECG report' in the report. STRICLTY 1 LINE ONLY. preserve as many words from the input prompt relavant to the ECG as possible in response this is to get good bleu score"
#     response = get_model_response(user_prompt, system_prompt)
#     return response

def generate_report(row):
    """Generates a report using a refined prompt for consistency with reference format, adding relevant information."""
    
    # Generate the refined prompt
    user_prompt = generate_prompt(row)
    
    # System instruction for precise structure
    system_prompt = (
        "Strictly follow this format: | Diagnostic class: [diagnostic class], heart axis: [heart axis value], "
        "infarction status: [infarction status], rate: [rate value], rhythm: [rhythm type] |. "
        "Return only the diagnostic report, and ensure it follows this single-line format exactly, with no extra text."
    )
    
    # Call the LLM with the refined prompt and structured system instructions
    response = get_model_response(user_prompt, system_prompt)
    
    # Post-process response to ensure consistency and remove extra whitespace
    processed_response = response.strip().replace('\n', '')
    
    # Log the final generated report for debugging if needed
    print(f"[DEBUG] Final Generated Report: {processed_response}")
    
    return processed_response




df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
agg_df = pd.read_csv("scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
print(df.head())
print("Diagnostic Superclasses created")

output_df = pd.DataFrame(columns=['ecg_id', 'report'])


for index, row in df.iterrows():
    try:
        ecg_id = row['ecg_id']
        print(f"Processing ECG ID: {ecg_id}")
        prompt=generate_prompt(row)
        # print(prompt)
        report = generate_report(row)
        print(report)
        # print(report)
        new_row = pd.DataFrame({'ecg_id': [ecg_id], 'report': [report]})
        
        # Concatenate the new row with the existing output_df
        output_df = pd.concat([output_df, new_row], ignore_index=True)
        
        # Save the DataFrame after processing each row to avoid data loss
        output_df.to_csv('ecg_reports_output.csv', index=False)
        
        # Log the progress
        logging.info(f"Processed ECG ID: {ecg_id}")
    except Exception as e:
        logging.error(f"Error processing ECG ID {ecg_id}: {e}")
