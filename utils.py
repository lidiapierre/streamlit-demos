import base64
import json
import os
import openai
import pickle

import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()


def get_llm():
    if os.getenv("LLM") == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI()
    else:
        return HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})


def get_embeddings():
    if os.getenv("EMBEDDINGS") == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings()
    else:
        embedding_model_name = os.environ.get('HF_EMBEDDING_MODEL_NAME')
        return HuggingFaceEmbeddings(model_name=embedding_model_name)


def get_text_splitter():
    if os.getenv("CHUNK_SIZE"):
        chunk_size = os.getenv("CHUNK_SIZE")
    else:
        chunk_size = 1000
    if os.getenv("CHUNK_OVERLAP"):
        chunk_overlap = os.getenv("CHUNK_OVERLAP")
    else:
        chunk_overlap = 200
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_vector_store_from_docs(docs, store_name=None):
    if store_name and os.path.exists(f"persist/{store_name}.pkl"):
        with open(f"persist/{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, get_embeddings())
        if store_name:
            with open(f"persist/{store_name}.pkl", "wb") as f:
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


def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "no": "2",
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "no": "3",
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}


def get_table_data(quiz_str):
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
