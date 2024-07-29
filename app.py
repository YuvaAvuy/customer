import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import nltk
import re
import string
import requests
from io import StringIO

# Downloading NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Defining the text cleaning function
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', re.sub('<.*?>+', '', text))
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Function to load dataset from Google Drive or Dropbox
@st.cache
def load_data():
    url = "https://drive.google.com/file/d/1w21XNPwb7x0lPpVHhMuTWHxjRZngRetN/view?usp=sharing"  # Update with your Google Drive file ID or Dropbox link
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    data = data.drop("Unnamed: 0", axis=1)
    data = data.dropna()
    data["Consumer complaint narrative"] = data["Consumer complaint narrative"].apply(clean)
    data = data[["Consumer complaint narrative", "Product"]]
    return data

data = load_data()
X = data["Consumer complaint narrative"]
y = data["Product"]

# Vectorizing the text data
cv = CountVectorizer()
X_vectorized = cv.fit_transform(X)

# Training the SGD classifier
sgdmodel = SGDClassifier()
sgdmodel.fit(X_vectorized, y)

# Streamlit app
st.title("Consumer Complaint Classification")

st.write("Enter a consumer complaint narrative below to predict the product category.")

user_input = st.text_area("Consumer Complaint Narrative")

if st.button("Predict"):
    if user_input:
        cleaned_input = clean(user_input)
        input_vectorized = cv.transform([cleaned_input])
        prediction = sgdmodel.predict(input_vectorized)
        st.write(f"Predicted Product: {prediction[0]}")
    else:
        st.write("Please enter a complaint narrative.")
