import streamlit as st
from langchain.chains.question_answering import load_qa_chain

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
        raw_text = get_pdf_text(pdf)
        store_name = pdf.name[:-4]
        vector_store = get_vector_store_from_text(store_name, raw_text)
        query = st.text_input("What is your question ?")

        if query:
            docs = vector_store.similarity_search(query)
            chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
            st.write(chain.run(input_documents=docs, question=query))


if __name__ == '__main__':
    main()
