import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import requests
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import nltk
import re
from nltk.corpus import stopwords
import string

# Downloading NLTK data
nltk.download('stopwords')

# Function to download and extract ZIP file
def download_and_extract_zip(url):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("dataset")
        # List the files in the zip to find the CSV
        return z.namelist()

# URL to the zipped dataset on GitHub
zip_url = ""
file_list = download_and_extract_zip(zip_url)

# Assuming the CSV is the first file in the zip
csv_file_path = "dataset/" + file_list[0]

# Load the dataset
data = pd.read_csv(csv_file_path)

# Drop the unnamed column
data = data.drop("Unnamed: 0", axis=1)

# Checking for null values
data = data.dropna()

# Defining the text cleaning function
stemmer = nltk.SnowballStemmer("english")
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

# Applying the cleaning function
data["Consumer complaint narrative"] = data["Consumer complaint narrative"].apply(clean)

# Selecting the relevant columns
data = data[["Consumer complaint narrative", "Product"]]

# Converting data to numpy arrays
x = np.array(data["Consumer complaint narrative"])
y = np.array(data["Product"])

# Vectorizing the text data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training the SGD classifier
sgdmodel = SGDClassifier()
sgdmodel.fit(X_train, y_train)

# Streamlit app layout
st.title("Consumer Complaint Classification")

user_input = st.text_area("Enter a Consumer Complaint:")

if st.button("Predict"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = sgdmodel.predict(data)
        st.write("Predicted Product:", output[0])
    else:
        st.write("Please enter a complaint.")
