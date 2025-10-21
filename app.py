import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps =  PorterStemmer()

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://github.com/ayshaimamx/spam_classification/raw/main/finalbg.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True
)

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

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message in question")
if st.button('Predict'):
# Steps for the model's working:
#     1) preprocess
    transformed_sms = transform_text(input_sms)
    #     2) vectorize
    vector_input = tfidf.transform([transformed_sms])
    #     3) predict
    result = model.predict(vector_input)
    #     4) display
    if result == 1:
        st.header("The given message is Spam")
    else:
        st.header("The given message is Not Spam")




