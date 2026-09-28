import streamlit as st
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------------
# FAQ DATA
# -----------------------------

faqs = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level programming language used for web development, AI, data science and automation."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence is a technology that enables computers to perform tasks that normally require human intelligence."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions."
    },
    {
        "question": "What is HTML?",
        "answer": "HTML stands for HyperText Markup Language and is used to create the structure of web pages."
    },
    {
        "question": "What is CSS?",
        "answer": "CSS stands for Cascading Style Sheets and is used to style and design web pages."
    },
    {
        "question": "What is JavaScript?",
        "answer": "JavaScript is a programming language used to make web pages interactive and dynamic."
    },
    {
        "question": "What is a database?",
        "answer": "A database is an organized collection of data that can be stored, managed and retrieved."
    },
    {
        "question": "What is SQL?",
        "answer": "SQL stands for Structured Query Language and is used to communicate with relational databases."
    },
    {
        "question": "What is React?",
        "answer": "React is a JavaScript library used to build user interfaces, especially for web applications."
    },
    {
        "question": "What is Django?",
        "answer": "Django is a Python web framework used to build secure and scalable web applications."
    }
]


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stop words
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Prepare FAQ questions
questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]


# -----------------------------
# TF-IDF MODEL
# -----------------------------

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(questions)


# -----------------------------
# CHATBOT FUNCTION
# -----------------------------

def get_answer(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity.argmax()

    best_score = similarity[0][best_match]

    # Minimum similarity threshold
    if best_score < 0.20:
        return "Sorry, I don't know the answer to that question."

    return faqs[best_match]["answer"]


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 FAQ Chatbot")

st.write(
    "Ask a question about Python, AI, Machine Learning, "
    "HTML, CSS, JavaScript, React, SQL or Django."
)

user_question = st.text_input(
    "Enter your question:"
)


if st.button("Ask"):

    if user_question.strip():

        answer = get_answer(user_question)

        st.subheader("Chatbot Response")

        st.success(answer)

    else:
        st.warning("Please enter a question.")
