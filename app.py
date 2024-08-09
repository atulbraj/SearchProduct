import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    st.title("Product Similarity Finder")

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
    query = st.text_input("Enter a product query:")

    if query:
        # Find similar products
        similar_products = find_similar_products_tfidf(query, vectorizer, tfidf_matrix, df, top_n=5)
        
        if not similar_products.empty:
            st.write("Similar Products:")
            st.write(similar_products)
        else:
            st.write("No similar products found.")

if __name__ == "__main__":
    main()


# import streamlit as st
# import pandas as pd
# import re
# import nltk
# from nltk import word_tokenize, WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')

# # Load Data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
#     return df

# df = load_data()

# # Initialize NLTK tools
# wordnet_lemmatizer = WordNetLemmatizer()

# # Define preprocessing functions
# def clean_text(text):
#     if isinstance(text, str):
#         text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
#         text = text.lower()  # Convert to lowercase
#     return text

# def lemmatize_text(text):
#     if isinstance(text, str):
#         return ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
#     return text

# def preprocess_text(text):
#     text = clean_text(text)
#     text = lemmatize_text(text)
#     return text

# # Combine relevant columns for the search index
# df['search_text'] = df['product_name'] + ' ' + df['brand'] + ' ' + df['product_category_tree']
# df['search_text'] = df['search_text'].apply(preprocess_text)
# df['search_text'] = df['search_text'].fillna('')

# # Initialize TF-IDF Vectorizer and fit_transform
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(df['search_text'])

# # Define function for retrieving products
# def retrieve_products(query, top_n=10):
#     query = preprocess_text(query)
#     query_vector = tfidf.transform([query])
#     cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
#     return df.iloc[related_docs_indices]

# # Streamlit App
# st.title("Product Search")

# # User input
# query = st.text_input("Enter your search query:", "")

# if query:
#     results_df = retrieve_products(query)
    
#     # Display search results
#     st.write(f"Top products for '{query}':")
    
#     # Use a grid layout for the product cards
#     num_cols = 3
#     cols = st.columns(num_cols)
    
#     for i, (index, row) in enumerate(results_df.iterrows()):
#         with cols[i % num_cols]:
#             # Extract and display the first image URL
#             image_urls = eval(row['image'])  # Convert string representation of list to list
#             first_image_url = image_urls[0] if isinstance(image_urls, list) and image_urls else "https://via.placeholder.com/200"
            
#             # Display product image
#             st.image(first_image_url, width=200, caption=row['product_name'])
            
#             # Display product name
#             st.write(f"**{row['product_name']}**")
            
#             # Display 'View Details' button
#             if st.button(f"View Details: {row['product_name']}", key=row['uniq_id']):
#                 st.write(f"**Product Name:** {row['product_name']}")
#                 st.write(f"**Brand:** {row['brand']}")
#                 st.write(f"**Retail Price:** {row['retail_price']}")
#                 st.write(f"**Discounted Price:** {row['discounted_price']}")
#                 st.write(f"**Rating:** {row['product_rating']}")
#                 st.write(f"**Description:** {row['description']}")
#                 st.write(f"**Specifications:** {row['product_specifications']}")
#                 st.image(first_image_url, width=400)
# else:
#     st.write("Please enter a search query to get started.")
