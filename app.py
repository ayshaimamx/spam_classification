import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps =  PorterStemmer()

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://lh3.googleusercontent.com/rd-gg-dl/AJfQ9KTmJJCZ6p49q7IrmzPa1XI0o1-Lyy5QlMjJYcQ7c9Ubdk5BFreiwmoTYf8sJwj1G0DhCMznEmBjK4luXP5zLw7OxEhGnmR-e4JAyPAQfznMojldULbHWWM9xSN3rJV6_4ZzNVWWPnOGupVQO4p6uUL2fauoPS_Tx-E5Sx-Ea7SiXTm6fZFcC0za6QW4_TCsoIz3TOiXSgT_iQPXgX2leZ3EfJrp76QUKAiWeQawCSB-06Z6mUZ1Ggf5i3hXXL6ln6zMvudcpgBl_aRBn0Mv4E_3qxiRpl_zDwywSKPuMbTZX6b_VpxH16Teh0Pg5nlOBSxm4_ugUh2Xqv6PuysdooQdzLhcgokBeeiEez8yhTZ1-2TYxcSQD1mKR6lsTfEPEtzEipaMjNluFpJAmDvvuHsgE_uRlHx29qAXuscrd4H_POh92nog3E6p_rB232CtQP1YyeKeUhE03XjV06d9B8oMGZjbjPUVBpw3MelZlH4ufFhzrEB5UiNzcrUnWLedCc2tANbUy8JsH0YpSRqzZuSnqr0NYKwIMkvqRfLtgdX2zas6_2r8SPW2Dt-qeCj6HbvfNOLiAtZAnz932LwEN3Xw_8aek4C4iVmhRfKj38Cz-cxhZi-FYL7q5h3LU6lWe8BZALkqV8EHfBGJapK7132_njJJ-xU4QcL40oJbRC4t_kVgWyay2A_dzUhJLDI9xmEMr07EerLVMnVGMA2XJN0gbm47ywm5KY68E-ECD_1Rr5AMkNkwi5zAfZPhZmeoGavU64a6Uh1350nE3u8DSvfUaVYLZc42QVQbLa6QProMiZeqxenwbk579k26B40lZAZSFrk9s5hWWNqhHeo85Kx5NEy_kxFpdJryOX0RIWSkbdq_gA2mJQ3LcAM2TezkMyJi9G606sLAJtmsc6SEkJ432PMGvv9JS1lMANO09qUB5Olezo25Q4GwNZx2ldRTPDPoHLrmn1tIZFQl2mYXFpBoEGu39U1CmoLXcJQnTm4w9NqSZQQzj1Av8xP9gERoskrvtMLLek6o_KlXja1z1evfQfGb8xQ9Pgvu-aaBYiLg92lJ1MBFhQkXN7G9YAGsb_4BNMu1ueplCsui5b8zYemRS6cRAS1t6fo1Msj-E3_5bxWygaCBgiQJPDxETrQbBzk7aXeVXDaDQbBo1H_lQFGxXRBouZG0QplJkNCovhkz_G7nuN3OMKl_lLQdizXRGUvW=s1024-rj");
        background-size: cover;eptual-word-cloud-for-for
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

