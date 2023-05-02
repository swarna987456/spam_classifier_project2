import streamlit as st
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()  # lower the text
    text = word_tokenize(text)  # convert to words
    y = []
    for i in text:  # remove special characters and punctuations
        if i.isalnum():
            y.append(i)

    text = y.copy()  # remove stopwords and special characters
    y.clear()
    # rint(text)
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()  # remove stem words
    y.clear()
    # rint(text)
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # return as a string


st.title('Email/SMS Spam Classifier')
input_sms = st.text_input('Enter the message')

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
