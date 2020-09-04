import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/IMDB_Dataset.csv")
x = df.iloc[:,0].values
y = df.iloc[:,1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x_train,y_train)

st.title("Movie Review Sentiment Analyzer")
review= st.text_area("Enter Review","Type Here . .")
y_pred = text_model.predict([review])
st.write("the sentiment of the review is ",y_pred)