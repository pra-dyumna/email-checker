import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
genai.configure(api_key="AIzaSyCMmroQORvijxS66b4ECA-1h6OgseWqezw")
model = genai.GenerativeModel("gemini-1.5-flash")

# Step 1: Load CSV Data
file_path = "interview_email_data.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Step 2: Preprocess and Analyze Dataset
# The dataset is assumed to have columns: 'Input Email' and 'User Reply'
email_response_pairs = list(zip(data["Input Email"], data["User Reply"]))

# Pre-compute TF-IDF embeddings for the email dataset
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([email for email, _ in email_response_pairs])

# Function to retrieve a similar reply based on a 90% match threshold
def retrieve_similar_reply(input_email, email_response_pairs):
    # Compute TF-IDF for the input email
    input_vector = vectorizer.transform([input_email])
    # Calculate cosine similarities
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    # Get the index of the most similar email
    best_match_idx = similarities.argmax()
    best_similarity = similarities[best_match_idx]
    # Check if similarity exceeds 90% (0.9)
    if best_similarity >= 0.9:
        return email_response_pairs[best_match_idx][1]  # Return the corresponding reply
    return None  # No match exceeds the threshold

# Step 3: RAG: Generate Response Grounded in Dataset
def generate_reply(input_email):
    # Retrieve the most similar reply from the dataset
    similar_reply = retrieve_similar_reply(input_email, email_response_pairs)
    
    # if similar_reply:
    #     # Step 4: Use Gemini API to format response based on the retrieved tone
    #     prompt = f"""Analyze the following user reply for tone and format: "{similar_reply}".
    #     Now, generate a reply for this new email: "{input_email}" in the same tone and format."""
    #     response = model.generate_content(prompt)
    #     return response.text
    # else:
    #     # Gracefully handle out-of-context queries
    #     return "Hello, Thank you for reaching out. Unfortunately, we do not have the requested information at this time. Please provide more details or let us know how we can assist further."
    if similar_reply:
        # Update the prompt to explicitly generate only the reply
        prompt = f"""The following is a reply example: "{similar_reply}".
        Based on this tone and format, write a concise and professional response to this email: "{input_email}".
        Only return the email response."""
        response = model.generate_content(prompt)
        return response.text.strip()  # Ensure clean output without extra spaces
    else:
        # Gracefully handle out-of-context queries
        return "Hello, Thank you for reaching out. Unfortunately, we do not have the requested information at this time. Please provide more details or let us know how we can assist further."
# Streamlit UI
st.title("Email Query Reply Generator")

# User input for unseen email
input_email = st.text_area("Enter your email query:", height=150)

# Generate reply button
if st.button("Generate Reply"):
    if input_email.strip():
        reply = generate_reply(input_email.strip())
        st.markdown(f"Generated Reply:")
        st.write(reply)
    else:
        st.error("Please enter a valid email query.")