import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from utils import add_bg_from_local
from htmlTemplates import css, user_template, bot_template # TODO is there better streamlit feature ???

import os

from dotenv import load_dotenv
load_dotenv()

# TODO refactor : chat with one PDFs or multiple PDFs or URLs

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

urls = [
    "https://medium.com/better-programming/a-comprehensive-guide-to-mlops-infrastructure-as-code-iac-ef4c97742351",
    "https://medium.com/d-one/building-an-end-to-end-mlops-pipeline-using-aws-sagemaker-c2aa1ebfaa5b",
    "https://towardsdatascience.com/why-should-data-scientists-adopt-machine-learning-ml-pipelines-8fc5e24920dd"
]


def handle_user_input(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def get_conversation_chain(vector_store): # TODO move to utils
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    llm = OpenAI(temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding_model)


def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


def load_urls():
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()


def main():
    st.set_page_config(page_title="Chatbot for our Own Website", page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)
    add_bg_from_local("images/abstract_5.jpg")
    st.header("Chat with your website 💬", )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    user_query = st.text_input("What is your question ?")
    if user_query:
        handle_user_input(user_query)

    # st.markdown("""
    # <style>
    #     [data-testid=stSidebar] {
    #         background-color: #ff000050;
    #     }
    # </style>
    # """, unsafe_allow_html=True)
    with st.sidebar:
        st.title("LLM Chat app using Langchain")
        st.markdown(f"""
        * [A comprehensive guide to MLOps Infrastructure As Code]({urls[0]})
        * [Building an End-to-end MLOps pipeline using AWS Sagemaker]({urls[1]})
        * [Why should Data Scientists adopt ML Pipelines]({urls[2]})
        """)

        if st.button("Start"):
            with st.spinner("Processing"):
                data = load_urls()
                chunks = get_chunks(data)
                vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Completed")


if __name__ == '__main__':
    main()
