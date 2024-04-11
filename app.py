


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained model and the TF-IDF vectorizer
nb_model = joblib.load('https://github.com/nzaki02/ML/blob/main/naive_bayes_model.joblib')
tfidf_vectorizer = joblib.load('https://github.com/nzaki02/ML/blob/main/tfidf_vectorizer.joblib')

def clean_text(text):
    # Implement your text cleaning process here
    return text.lower()

def predict_sentiment(review):
    # Preprocess and vectorize the review
    review_vectorized = tfidf_vectorizer.transform([review])
    # Predict sentiment
    sentiment = nb_model.predict(review_vectorized)
    # Get predicted probabilities for each class
    probabilities = nb_model.predict_proba(review_vectorized)
    # Get the probability associated with the predicted class
    confidence_score = max(probabilities[0])
    return 'Positive' if sentiment == 1 else 'Negative', confidence_score

def process_file(uploaded_file):
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Ensure there is a "Review" column in the uploaded file
    if 'Review' not in df.columns:
        st.error("CSV file must contain a 'Review' column.")
        return None
    
    # Apply the sentiment prediction and confidence score calculation to each review
    results = df['Review'].apply(lambda review: predict_sentiment(clean_text(review)))
    
    # Split the results into separate columns and add them to the original DataFrame
    df['Sentiment'], df['Confidence Score'] = zip(*results)
    
    # Calculate the percentage of positive vs negative sentiments
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    sentiment_counts = sentiment_counts.round(2)
    
    # Display the sentiment distribution as a pie chart without slice labels
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size as needed
    pie_wedges = sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, labels=None, 
                                           colors=['#4CAF50', '#F44336'], ax=ax, textprops={'fontsize': 12})
    plt.title('Sentiment Distribution', fontsize=8)  # Adjust title font size
    
    # Add a legend with custom labels
    ax.legend(pie_wedges, labels=['Positive', 'Negative'], title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    st.pyplot(fig)
    
    return df

    return df



# Streamlit app interface for single review submission
st.title('Restaurant Review Sentiment Analysis')
user_review = st.text_area("Enter a restaurant review:", key="user_review_input")

if st.button('Predict Sentiment', key='predict_sentiment_button'):
    cleaned_review = clean_text(user_review)
    sentiment, confidence = predict_sentiment(cleaned_review)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Confidence Score: {confidence:.2f}')


# Batch submission interface
st.header("Batch Submission")
uploaded_file = st.file_uploader("Choose a CSV file containing 'Review' column", type="csv")
if uploaded_file is not None:
    processed_data = process_file(uploaded_file)
    if processed_data is not None:
        st.write("Prediction Results:")
        st.dataframe(processed_data[['Review', 'Sentiment', 'Confidence Score']])
