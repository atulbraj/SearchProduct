import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

# Load the DataFrame and Word2Vec model
df = pd.read_pickle('products.pkl')
model = Word2Vec.load('word2vec_model.bin')

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    return tokens

# Define the embedding function
def get_embedding(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Define functions to get query embedding and find similar products
def get_query_embedding(query, model):
    tokens = preprocess_text(query)
    return get_embedding(tokens, model)

def find_similar_products(query, df, model, top_n=5):
    query_embedding = get_query_embedding(query, model)
    
    # Calculate cosine similarities
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    
    # Get top_n similar products
    similar_products = df.sort_values(by='similarity', ascending=False).head(top_n)
    
    # Return relevant columns
    return similar_products[['pid', 'product_name', 'description', 'similarity']]

# Streamlit UI
st.title('Product Recommendation System')

# Input box for the query
query = st.text_input('Enter a product query:', 'samsung phone')

# Button to get recommendations
if st.button('Find Similar Products'):
    similar_products = find_similar_products(query, df, model, top_n=5)
    
    # Display the results
    if not similar_products.empty:
        st.write('### Top Similar Products:')
        st.dataframe(similar_products)
    else:
        st.write('No similar products found.')
