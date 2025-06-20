import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

port_stem = PorterStemmer()

pickle_in = open("trained_model.sav","rb")
classifier=pickle.load(pickle_in)
nltk.download('stopwords')
vectorizer = TfidfVectorizer()



def load_training_data(filename):
    # Load the training data from a file
    with open(filename, 'rb') as file:
        X_train = pickle.load(file)
    return X_train


def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content


def predict_the_tweet(tweet):
    stemmed_tweet = stemming(tweet)
    X_train = load_training_data("vectorization_fit_data.pkl")
    vectorizer.fit(X_train)
    vectorized_tweet = vectorizer.transform([stemmed_tweet])
    prediction = classifier.predict(vectorized_tweet)
    
    return prediction
    

def main():
    st.title("Tweeter Sentiment Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Tweeter Sentiment Analysis ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    tweet = st.text_input("Tweet","Type Here")
   
    result=""
    if st.button("Predict"):
        result=predict_the_tweet(tweet)
        if(result[0] == 0):
            st.success('The output is negative Tweet {}')
            print("negative Tweet")
        else:
            st.success('The output is Positive Tweet {}')
            print("Positive Tweet")
    
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
       
        

if __name__=='__main__':
    main()


