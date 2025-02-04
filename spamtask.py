import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from wordcloud import  WordCloud
import re
import nltk
from nltk.tokenize import  word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
data = data[["class", "message"]]
stop_words=set(stopwords.words('english'))
lemma=WordNetLemmatizer()
def clean(text):
    if text is None:
        return ''
    text=re.sub(r'https\S+','',text)
    text=re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    text=word_tokenize(text)
    text=[lemma.lemmatize(word=w,pos='v')for w in text if w not in stop_words  if len(w)>2]
    text=' '.join(text)
    return text
data['cleanmessage']=data['message'].apply(clean)
data=data.drop('message',axis=1)
x = np.array(data["cleanmessage"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.title("Spam Detection System")
def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(*a)
spamdetection()
st.write(f'Accuracy is  {accuracy * 100:.2f}%')
wordcloud=WordCloud(width=800,height=300,background_color='black').generate(''.join(data['cleanmessage']))
def word_cloud():
    fig,ax=plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud,interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
st.write(" This is the wordcloud generated from  the CleanMessage")
word_cloud()