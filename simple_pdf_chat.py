import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from htmlTemplates import css, bot_template, user_template
from utils import add_bg_from_local, get_llm, get_pdf_text, get_vector_store_from_text

llm = get_llm()

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


def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config("Chat with multiple PDFs", page_icon=":books:")
    add_bg_from_local("images/abstract_4.jpg")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with your PDFs 💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question from your documents")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.header("Your Documents")
        st.markdown('* Powered by [LangChain](https://python.langchain.com/docs/get_started/introduction) 🦜⛓️')
        pdf_docs = st.file_uploader("Upload your PDF files here then click on Process", accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                vector_store = get_vector_store_from_text(raw_text)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Done!")


if __name__ == '__main__':
    main()
