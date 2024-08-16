import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    return df

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to find similar products using TF-IDF
def find_similar_products_tfidf(query, vectorizer, tfidf_matrix, df, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Add similarity scores to dataframe
    df['similarity'] = similarities

    # Filter products with similarity greater than 0.30
    similar_products = df[df['similarity'] > 0.30].sort_values(by='similarity', ascending=False).head(top_n)

    return similar_products[['pid', 'product_name', 'description', 'similarity']]

# Main Streamlit app
def main():
    # App title and creator details
    st.title("Product Search Retrieval System")
    st.write("---")  # Divider

    # Load data
    df = load_data()

    # Select relevant columns
    rel_cols = ["product_name", "product_category_tree", "description", "product_rating", 
                "overall_rating", "brand", "product_specifications"]

    # Combine relevant content into a single column
    df['relevant_content'] = df[rel_cols].fillna('').agg(' '.join, axis=1).apply(preprocess_text)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['relevant_content'])

    # User input for the query
    st.subheader("Find Relevant Products")
    query = st.text_input("Enter a product query:")

    if query:
        # Find similar products
        similar_products = find_similar_products_tfidf(query, vectorizer, tfidf_matrix, df, top_n=5)
        
        if not similar_products.empty:
            st.write("**Similar Products:**")
            st.dataframe(similar_products)
        else:
            st.warning("No similar products found.")
    
    # Footer
    st.write("---")
    st.markdown("""
        **Creators**: *Team: Walmartians*  
    """)
    st.caption("Developed with (💻 && 🧠)")

if __name__ == "__main__":
    main()
