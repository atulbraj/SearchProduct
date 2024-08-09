# import streamlit as st
# import pandas as pd
# import re
# import nltk
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load Data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
#     return df

# df = load_data()

# # Initialize NLTK tools
# stop_words = set(stopwords.words('english'))

# # Define preprocessing functions
# def clean_text(text):
#     if isinstance(text, str):
#         text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
#         text = text.lower()  # Convert to lowercase
#     return text

# def preprocess_text(text):
#     text = clean_text(text)
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in stop_words]
#     return tokens

# # Combine relevant columns for the search index
# df['search_text'] = df['product_name'] + ' ' + df['brand'] + ' ' + df['product_category_tree']
# df['tokens'] = df['search_text'].apply(preprocess_text)

# # Train Word2Vec model
# model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, sg=0)

# def get_embedding(tokens, model):
#     vectors = [model.wv[token] for token in tokens if token in model.wv]
#     if len(vectors) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(vectors, axis=0)

# # Generate embeddings for each product
# df['embedding'] = df['tokens'].apply(lambda x: get_embedding(x, model))

# # Define function for retrieving products
# def retrieve_products(query, top_n=10, threshold=0.3):
#     query_tokens = preprocess_text(query)
#     query_embedding = get_embedding(query_tokens, model)
#     cosine_similarities = cosine_similarity([query_embedding], list(df['embedding'].values)).flatten()
    
#     # Apply threshold to filter out less similar products
#     filtered_indices = np.where(cosine_similarities >= threshold)[0]
    
#     if len(filtered_indices) == 0:
#         return pd.DataFrame()  # Return an empty DataFrame if no products meet the threshold
    
#     related_docs_indices = cosine_similarities.argsort()[::-1][:top_n]
#     return df.iloc[related_docs_indices]

# # Streamlit App
# st.title("Product Search with Word2Vec")

# # User input
# query = st.text_input("Enter your search query:", "")

# if query:
#     results_df = retrieve_products(query)
    
#     if not results_df.empty:
#         st.write(f"Top products for '{query}':")
        
#         # Use a grid layout for the product cards
#         num_cols = 3
#         cols = st.columns(num_cols)
        
#         for i, (index, row) in enumerate(results_df.iterrows()):
#             with cols[i % num_cols]:
#                 # Extract and display the first image URL
#                 image_urls = eval(row['image'])  # Convert string representation of list to list
#                 first_image_url = image_urls[0] if isinstance(image_urls, list) and image_urls else "https://via.placeholder.com/200"
                
#                 # Display product image
#                 st.image(first_image_url, width=200, caption=row['product_name'])
                
#                 # Display product name
#                 st.write(f"**{row['product_name']}**")
                
#                 # Display 'View Details' button
#                 if st.button(f"View Details: {row['product_name']}", key=row['uniq_id']):
#                     st.write(f"**Product Name:** {row['product_name']}")
#                     st.write(f"**Brand:** {row['brand']}")
#                     st.write(f"**Retail Price:** {row['retail_price']}")
#                     st.write(f"**Discounted Price:** {row['discounted_price']}")
#                     st.write(f"**Rating:** {row['product_rating']}")
#                     st.write(f"**Description:** {row['description']}")
#                     st.write(f"**Specifications:** {row['product_specifications']}")
#                     st.image(first_image_url, width=400)
#     else:
#         st.write("No similar products found. Try a different query.")
# else:
#     st.write("Please enter a search query to get started.")


import streamlit as st
import pandas as pd
import re
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    return df

df = load_data()

# Initialize NLTK tools
wordnet_lemmatizer = WordNetLemmatizer()

# Define preprocessing functions
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
    return text

def lemmatize_text(text):
    if isinstance(text, str):
        return ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
    return text

def preprocess_text(text):
    text = clean_text(text)
    text = lemmatize_text(text)
    return text

# Combine relevant columns for the search index
df['search_text'] = df['product_name'] + ' ' + df['brand'] + ' ' + df['product_category_tree']
df['search_text'] = df['search_text'].apply(preprocess_text)
df['search_text'] = df['search_text'].fillna('')

# Initialize TF-IDF Vectorizer and fit_transform
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['search_text'])

# Define function for retrieving products
def retrieve_products(query, top_n=10):
    query = preprocess_text(query)
    query_vector = tfidf.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    return df.iloc[related_docs_indices]

# Streamlit App
st.title("Product Search")

# User input
query = st.text_input("Enter your search query:", "")

if query:
    results_df = retrieve_products(query)
    
    # Display search results
    st.write(f"Top products for '{query}':")
    
    # Use a grid layout for the product cards
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, (index, row) in enumerate(results_df.iterrows()):
        with cols[i % num_cols]:
            # Extract and display the first image URL
            image_urls = eval(row['image'])  # Convert string representation of list to list
            first_image_url = image_urls[0] if isinstance(image_urls, list) and image_urls else "https://via.placeholder.com/200"
            
            # Display product image
            st.image(first_image_url, width=200, caption=row['product_name'])
            
            # Display product name
            st.write(f"**{row['product_name']}**")
            
            # Display 'View Details' button
            if st.button(f"View Details: {row['product_name']}", key=row['uniq_id']):
                st.write(f"**Product Name:** {row['product_name']}")
                st.write(f"**Brand:** {row['brand']}")
                st.write(f"**Retail Price:** {row['retail_price']}")
                st.write(f"**Discounted Price:** {row['discounted_price']}")
                st.write(f"**Rating:** {row['product_rating']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Specifications:** {row['product_specifications']}")
                st.image(first_image_url, width=400)
else:
    st.write("Please enter a search query to get started.")
