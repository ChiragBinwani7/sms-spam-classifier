import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
ar = np.array([1,2,3])
pd.DataFrame({"col" : ar})

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import pickle
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('ml_model.pkl','rb'))


def text_transformation(text):
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

    return (" ").join(y)




st.title("Sms Spam Classifier")
sms = st.text_area("Enter Your Message")

if st.button("predict"):
    transformed_sms = text_transformation(sms)
    vector_sms = tfidf.transform([transformed_sms])
    result = model.predict(vector_sms)[0]
    if result == 1:
        st.header("Spam!!!")
    else:
        st.header("Not Spam")