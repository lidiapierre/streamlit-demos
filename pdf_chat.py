import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import pickle

from utils import add_bg_from_local
from PyPDF2 import PdfReader
import os

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

textsplitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

advanced_prompt = """
You are a Bot assistant answering any questions about documents.
You are given a question and a set of documents.
If the user's question requires you to provide specific information from the documents, give your answer based only 
on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't 
find the answer in the documentation and propose him to rephrase his query with more details.
Use bullet points if you have to make a list, only if necessary.

QUESTION: {question}

DOCUMENTS:
=========
{context}
=========
"""


def main():
    add_bg_from_local("images/abstract_4.jpg")
    st.header("Chat with your PDF")
    st.sidebar.title("LLM Chat app using LangChain")
    st.sidebar.markdown('''
    This is a LLM powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    ''')
    st.sidebar.write("Upload a PDF file and start asking questions!")

    pdf = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf is not None:
        reader = PdfReader(pdf)
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        raw_text = " ".join(raw_text.split())

        texts = textsplitter.split_text(raw_text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            vector_store = FAISS.from_texts(texts, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        query = st.text_input("What is your question ?")

        if query:
            docs = vector_store.similarity_search(query)
            chain = load_qa_chain(OpenAI(), chain_type='stuff', verbose=True)
            response = chain.run(input_documents=docs, question=query)

            st.write(response)


if __name__ == '__main__':
    main()
