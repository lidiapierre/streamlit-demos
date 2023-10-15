import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.memory import ConversationBufferMemory

from htmlTemplates import css, bot_template, user_template
from utils import add_bg_from_local, get_llm, get_vector_store_from_docs

llm = get_llm()


def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def chat_conversation(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config("Chat with your own data", page_icon=":books:")
    add_bg_from_local("images/abstract_4.jpg")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with your data 💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your data")
    if user_question:
        chat_conversation(user_question)

    with st.sidebar:
        st.header("Your Documents and Links")
        st.markdown('Powered by [LangChain](https://python.langchain.com/docs/get_started/introduction) 🦜⛓️')

        pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True)
        urls_text = st.text_area("Your URLs", placeholder="https://link1\nhttps://link2\nhttps://link3")

        if st.button('Process'):
            with st.spinner("Processing"):
                data = []
                if urls_text:
                    urls = urls_text.split()
                    loader = UnstructuredURLLoader(urls=urls)
                    data.extend(loader.load())

                if pdf_docs:
                    loader = UnstructuredPDFLoader(pdf_docs)
                    data.extend(loader.load())

                if data:
                    vector_store = get_vector_store_from_docs(data)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Done!")
                else:
                    st.write("No input")


if __name__ == '__main__':
    main()
