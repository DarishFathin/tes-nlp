import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ========== Load & Preprocess ==========
@st.cache_data
def load_data():
    df = pd.read_csv("Mental_Health_FAQ_ID.csv")
    df = df[["Pertanyaan_ID", "Jawaban_ID"]].dropna()
    df = df[~df["Jawaban_ID"].str.contains("Terjemahan gagal", case=False, na=False)]
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
st.set_page_config(page_title="Chatbot Kesehatan Mental", page_icon="🧠")
st.title("🧠 Chatbot Kesehatan Mental")
st.markdown("Tanyakan hal-hal seputar kesehatan mental, dan saya akan menjawab berdasarkan FAQ!")

# Contoh pertanyaan
contoh_pertanyaan = [
    "Apa itu depresi?",
    "Bagaimana cara mengatasi rasa cemas?",
    "Apa perbedaan stres dan kecemasan?",
    "Saya susah tidur, apakah itu normal?",
    "Kapan saya harus ke psikolog?",
]

selected = st.selectbox("💡 Pilih contoh pertanyaan:", options=[""] + contoh_pertanyaan)

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if selected:
    st.session_state["user_input"] = selected

user_input = st.text_input("Atau ketik pertanyaanmu di sini:", value=st.session_state["user_input"])

if user_input:
    cleaned = bersihkan_teks(user_input)
    response = model.predict([cleaned])[0]
    st.markdown(f"**🤖 Bot:** {response}")
