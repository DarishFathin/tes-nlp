import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st

# ========== Load & Preprocess ==========
@st.cache_data
def load_data():
    df = pd.read_csv("Mental_Health_FAQ_ID.csv")
    df = df[["Pertanyaan_ID", "Jawaban_ID"]].dropna()
    df["pertanyaan_bersih"] = df["Pertanyaan_ID"].apply(bersihkan_teks)
    return df

def bersihkan_teks(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

df = load_data()

# ========== Train Model ==========
X = df["pertanyaan_bersih"]
y = df["Jawaban_ID"]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])
model.fit(X, y)

# ========== Streamlit Interface ==========
st.title("🧠 Chatbot Kesehatan Mental")
st.markdown("Tanyakan hal-hal seputar kesehatan mental, dan saya akan membantumu menjawab!")

user_input = st.text_input("Apa yang ingin Anda tanyakan? (kecemasan/depresi/tidur/konseling/hubungan sosial)")

if user_input:
    cleaned = bersihkan_teks(user_input)
    response = model.predict([cleaned])[0]
    st.markdown(f"**🤖 Bot:** {response}")
