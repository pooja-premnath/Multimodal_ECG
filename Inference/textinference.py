import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ECGSimilarityModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the ECG Similarity Model with a pre-trained embedding model.
        """
        logging.info("Loading the embedding model.")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.ecg_ids = None

    def load_data(self, csv_path):
        """
        Loads and preprocesses the data from CSV file.
        """
        logging.info(f"Loading data from {csv_path}.")
        try:
            self.df = pd.read_csv(csv_path)
            if 'report' not in self.df.columns or 'ecg_id' not in self.df.columns:
                raise ValueError("CSV must contain 'report' and 'ecg_id' columns.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise
        logging.info("Data loaded successfully.")

    def compute_embeddings(self):
        """
        Computes and stores embeddings for each report in the dataset.
        """
        logging.info("Generating embeddings for reports.")
        reports = self.df['report'].tolist()
        self.embeddings = self.model.encode(reports, convert_to_tensor=True)
        self.ecg_ids = self.df['ecg_id'].tolist()
        logging.info("Embeddings generated successfully.")

    def save_model(self, embeddings_path='ecg_embeddings.pkl', model_path='embedding_model'):
        """
        Saves embeddings and the model to disk.
        """
        logging.info("Saving embeddings and model.")
        try:
            with open(embeddings_path, 'wb') as f:
                pickle.dump({'ecg_ids': self.ecg_ids, 'embeddings': self.embeddings}, f)
            self.model.save(model_path)
        except Exception as e:
            logging.error(f"Error saving model or embeddings: {e}")
            raise
        logging.info("Model and embeddings saved successfully.")

    def load_model(self, embeddings_path='ecg_embeddings.pkl', model_path='embedding_model'):
        """
        Loads the model and embeddings for inference.
        """
        logging.info("Loading embeddings and model for inference.")
        try:
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.ecg_ids = data['ecg_ids']
                self.embeddings = data['embeddings']
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            logging.error(f"Error loading model or embeddings: {e}")
            raise
        logging.info("Embeddings and model loaded successfully.")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def find_similar_reports(self, input_report, top_k=5):
        """
        Finds the top K most similar reports based on cosine similarity.
        """
        logging.info("Calculating similarity for the input report.")
        
        # Encode the input report and move to CPU
        input_embedding = self.model.encode([input_report], convert_to_tensor=True).cpu().numpy()
        
        # Ensure stored embeddings are also on the CPU and converted to NumPy arrays
        embeddings_cpu = self.embeddings.cpu().numpy()
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_embedding, embeddings_cpu)[0]
        
        # Sort and retrieve top K indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_reports = [(self.ecg_ids[i], similarities[i]) for i in top_k_indices]

        logging.info("Similarity calculation completed.")
        return similar_reports



# Example Usage
if __name__ == "__main__":
    model = ECGSimilarityModel()
    # model.load_data('ecg_reports_output.csv')
    # model.compute_embeddings()
    # model.save_model()  # Save for future inference

    # Load for inference
    model.load_model()
    input_report = "Normal sinus rhythm, no heart axis data available, diagnostic classification: NORM, no infarction detected"
    similar_reports = model.find_similar_reports(input_report)
    
    logging.info("Top 5 similar reports:")
    for ecg_id, score in similar_reports:
        logging.info(f"ECG ID: {ecg_id}, Similarity Score: {score:.4f}")

