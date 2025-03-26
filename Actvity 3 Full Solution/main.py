from preprocess import *          # Functions like get_input_data(), de_duplication(), noise_remover()
from embeddings import *          # Function get_tfidf_embd() for TF-IDF embeddings
from modelling.modelling import * # Function model_predict() for modeling
from modelling.data_model import *# Data class to hold features and labels
import random
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    """Load raw email data from CSV files."""
    df = get_input_data()  # Defined in preprocess.py
    return df

def preprocess_data(df):
    """Clean the data by removing duplicates and noise."""
    df = de_duplication(df)  # Remove duplicate entries
    df = noise_remover(df)   # Remove noisy data
    return df

def get_embeddings(df: pd.DataFrame):
    """Generate TF-IDF embeddings from text data."""
    X = get_tfidf_embd(df)  # From embeddings.py
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create a Data object with features and multi-label targets."""
    return Data(X, df)  # Data class (in data_model.py) now handles 'y2', 'y3', 'y4'

def perform_modelling(data: Data, df: pd.DataFrame, name):
    """Run the modeling process with the chained classifier."""
    model_predict(data, df, name)  # Updated in modelling.py for multi-label

if __name__ == '__main__':
    # Step 1: Load the raw data
    df = load_data()
    
    # Step 2: Preprocess the data
    df = preprocess_data(df)
    
    # Ensure text columns are strings (assuming Config defines column names)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Step 3: Group data by 'y1' (Type1)
    grouped_df = df.groupby(Config.GROUPED)
    
    # Step 4: Process each group
    for name, group_df in grouped_df:
        print(f"Processing group: {name}")
        
        # Generate embeddings for this group
        X, group_df = get_embeddings(group_df)
        
        # Create Data object (handles multi-labels internally)
        data = get_data_object(X, group_df)
        
        # Check if there’s enough data to model (optional check)
        if data.X_train is not None:
            # Run modeling with the chained multi-label approach
            perform_modelling(data, group_df, name)