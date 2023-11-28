
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Load the pre-trained MultinomialNB model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Debug prints
    print("Transformed Message:", transformed_sms)

    # Check if the model is fitted
    if hasattr(model, 'fit'):
        # Assuming you have X_train (messages) and y_train (labels) defined somewhere
        X_train = ["your", "training", "data"]  # Replace with your actual training data
        y_train = [0, 1, 0]  # Replace with your actual labels

        # Fit the MultinomialNB model with the training data
        X_train_transformed = tfidf.transform(X_train)
        model.fit(X_train_transformed, y_train)

        # Vectorize the input message
        vector_input = tfidf.transform([transformed_sms])

        # Debug prints
        print("Vectorized Input:", vector_input)

        # Make predictions
        result = model.predict(vector_input)

        # Debug prints
        print("Prediction Result:", result)

        # Display the result
        if result[0] == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Model not fitted. Please fit the model before making predictions.")
