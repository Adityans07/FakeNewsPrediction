import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
Ps = PorterStemmer()

import re
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

tfidf = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Fake news detector")

input_title = st.text_input("Enter title")
input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    transform_news = stemming(input_sms)
    transform_title = stemming(input_title)
    vector_input = tfidf.transform([transform_news,transform_title])
    result = model.predict(vector_input)[0]
    if result==1:
        st.header("Fake")
    else:
        st.header("Not fake")