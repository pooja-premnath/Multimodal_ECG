import pandas as pd
import ast
import logging
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


llm = Ollama(model="llama3.1")


df = pd.read_csv("ptbxl_database.csv")
loader = PDFPlumberLoader("The-ECG-Made-Easy-9th-Edition.pdf")
docs = loader.load()
text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)
embedder = HuggingFaceEmbeddings()

# Create the vector store 
vector = FAISS.from_documents(documents, embedder)

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )



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


def generate_prompt(row):
    diagnostic_superclass_mapping = {
        '[NORM]': 'Normal ECG',
        '[MI]': 'Myocardial Infarction',
        '[STTC]': 'ST/T Change',
        '[CD]': 'Conduction Disturbance',
        '[HYP]': 'Hypertrophy'
    }
    report = row['report']
    heart_axis = row['heart_axis'] if not pd.isnull(row['heart_axis']) else 'no heart axis data'
    infarction = "No infarction detected" if pd.isnull(row['infarction_stadium1']) else "Infarction detected"
    diagnostic_superclass_raw = row['diagnostic_superclass']
    

    prompt = (f"Generate a concise 1-line ECG report for a patient. "
              f"The report mentions: '{report}' Translate that to english first, heart axis as {heart_axis}, "
              f"diagnostic superclass: {diagnostic_superclass_raw}, and {infarction}. Generate only one line report, nothing else. ")
    
    return prompt


def generate_report(row):
    user_prompt = generate_prompt(row)
    system_prompt = "Strictly 1 line only. Generate only report, nothing else. Start main report with | and end with |. Do not include any other information. Do not use the phrase 'ECG report' in the report."
    response = get_model_response(user_prompt, system_prompt)
    return response





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
        report=qa(report)["result"]

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
