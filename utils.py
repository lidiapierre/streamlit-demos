import streamlit as st
import base64
import json
import traceback
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import pickle
from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()


def get_llm():
    if os.getenv("LLM") == "openai":
        return OpenAI()


def get_embeddings():
    if os.getenv("EMBEDDINGS") == "openai":
        return OpenAIEmbeddings()


def get_text_splitter():
    if os.getenv("CHUNK_SIZE"):
        chunk_size = os.getenv("CHUNK_SIZE")
    else:
        chunk_size = 1000
    if os.getenv("CHUNK_OVERLAP"):
        chunk_overlap = os.getenv("CHUNKCHUNK_OVERLAP_SIZE")
    else:
        chunk_overlap = 1000
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_pdf_text(file):
    reader = PdfReader(file)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    return " ".join(raw_text.split())


def get_vector_store_from_text(store_name, text):
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        text_splitter = get_text_splitter()
        texts = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(texts, get_embeddings())
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
        },
        "correct": "correct answer",
    },
}

def get_table_data(quiz_str):
    try:
        # convert the quiz from a str to dict
        quiz_dict = json.loads(quiz_str)
        quiz_table_data = []
        # Iterate over the quiz dictionary and extract the required information
        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
