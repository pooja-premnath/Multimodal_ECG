import streamlit as st
import pandas as pd
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import logging
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageTextSimilarityModel:
    def __init__(self, text_model_name='all-MiniLM-L6-v2', image_model_name='resnet50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.text_model = SentenceTransformer(text_model_name)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))
        self.image_model.to(self.device)
        
        self.index = None
        self.ecg_ids = None
        self.average_text_embedding = None
        
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        return preprocess(image).unsqueeze(0).to(self.device)

    def load_model(self, embeddings_path='faiss_embeddings.pkl', faiss_index_path='faiss_index.bin'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.ecg_ids = data['ecg_ids']
            self.average_text_embedding = data['average_text_embedding'].to(self.device)
        self.index = faiss.read_index(faiss_index_path)
        logging.info("Model and FAISS index loaded successfully.")

    def find_similar_reports(self, image, method='faiss', top_k=5):
        image_tensor = self.preprocess_image(image)
        image_embedding = self.image_model(image_tensor).squeeze().detach()
        input_embedding = torch.cat((image_embedding, self.average_text_embedding.detach())).cpu().numpy()
        
        if method == 'faiss':
            _, indices = self.index.search(np.array([input_embedding]), top_k)
            similar_reports = [(self.ecg_ids[i], i) for i in indices[0]]
        else:  # cosine
            similar_reports = []
            for i in range(self.index.ntotal):
                candidate_embedding = self.index.reconstruct(i)
                similarity_score = cosine_similarity([input_embedding], [candidate_embedding])[0][0]
                similar_reports.append((self.ecg_ids[i], similarity_score))
            similar_reports = sorted(similar_reports, key=lambda x: x[1], reverse=True)[:top_k]
            
        return similar_reports

def load_image_safely(image_path):
    """Safely load an image file, returning None if the file doesn't exist"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"Image not found: {image_path}")
            return None
    except Exception as e:
        st.warning(f"Error loading image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="ECG Report Similarity Finder", layout="wide")
    
    # Add custom CSS with updated styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .report-card {
            background: linear-gradient(135deg, #f6f8fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #e1e4e8;
        }
        .similarity-score {
            color: #2ea44f;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            background: rgba(46, 164, 79, 0.1);
            border-radius: 0.5rem;
            display: inline-block;
        }
        .report-title {
            color: #1a73e8;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .report-content {
            color: #24292e;
            font-size: 1rem;
            line-height: 1.6;
            padding: 0.5rem;
            background: white;
            border-radius: 0.5rem;
            border: 1px solid #e1e4e8;
        }
        .uploaded-image-card {
            background: white;
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border: 1px solid #e1e4e8;
        }
        .stExpander {
            border: none !important;
            box-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🫀 ECG Report Similarity Finder")
    st.markdown("Upload an ECG image to find similar reports from our database.")
    
    # Initialize model
    @st.cache_resource
    def load_similarity_model():
        model = ImageTextSimilarityModel()
        model.load_model()
        return model
    
    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('ecg_reports_outpu100_filtered.csv')
    
    try:
        model = load_similarity_model()
        df = load_data()
        
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            similarity_method = st.selectbox(
                "Similarity Method",
                ["FAISS (L2 Distance)", "Cosine Similarity"],
                index=0
            )
            top_k = st.slider("Number of similar reports", min_value=1, max_value=10, value=5)
            
            # Add image base path setting
            image_base_path = st.text_input(
                "Image Base Path",
                value="",
                help="Base path where your ECG images are stored"
            )
            
        # File uploader
        uploaded_file = st.file_uploader("Choose an ECG image file", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                with st.container():
                    st.markdown('<div class="uploaded-image-card">', unsafe_allow_html=True)
                    st.image(image, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Find similar reports
            method = 'faiss' if 'FAISS' in similarity_method else 'cosine'
            similar_reports = model.find_similar_reports(image, method=method, top_k=top_k)
            
            # Display results
            with col2:
                st.subheader("Similar Reports")
                for ecg_id, score in similar_reports:
                    with st.expander(f"ECG ID: {ecg_id}", expanded=True):
                        report_row = df[df['ecg_id'] == ecg_id].iloc[0]
                        
                        # Construct image path
                        image_path = os.path.join(image_base_path, f"{report_row['filename_lr']}.png")
                        
                        # Create two columns for image and report
                        img_col, text_col = st.columns(2)
                        
                        with img_col:
                            similar_image = load_image_safely(image_path)
                            if similar_image:
                                st.image(similar_image, use_column_width=True)
                        
                        with text_col:
                            st.markdown(f"""
                            <div class="report-card">
                                <div class="similarity-score">Similarity Score: {score:.4f}</div>
                                <div class="report-title">Report Details</div>
                                <div class="report-content">{report_row['report']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required files (model, index, and dataset) are in the correct location.")

if __name__ == "__main__":
    main()