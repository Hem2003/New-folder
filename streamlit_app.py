import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the image
try:
    image = Image.open('me2.png.jpg')
    st.image(image, caption='EMAIL')
except FileNotFoundError:
    st.write("Image not found!")

# Load pre-trained models and resources
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]
    return " ".join(filtered_tokens)

# Streamlit UI
st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From:", ["Via Email", "Via SMS", "Other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')
