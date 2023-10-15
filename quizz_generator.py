import json
import sys

import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper

from utils import add_bg_from_local, get_llm, RESPONSE_JSON

quizz_template = PromptTemplate(
    input_variables=['text', 'response_json'],
    template="""
Text:{text}
Given the above text, create a quiz of 5 multiple choice questions. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
### RESPONSE_JSON
{response_json}
"""
)

llm = get_llm()
wiki = WikipediaAPIWrapper()
summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff")

quiz_chain = LLMChain(llm=llm, prompt=quizz_template, output_key="quizz", verbose=True)


def main():
    st.title("Quizz Generator 📚")
    add_bg_from_local("images/abstract_1.jpg")

    with st.form("user_inputs"):
        topic = st.text_input("Enter the topic you want to be tested on")
        url = st.text_input("Optional: enter a link to the course content")
        button = st.form_submit_button("Create Quizz")

        if button:
            docs = []
            if not topic:
                sys.exit("Missing topic")
            wiki_search = wiki.load(topic)
            docs.extend(wiki_search)
            if url:
                url_data = UnstructuredURLLoader([url], mode="single").load()[0]
                st.write(url_data.page_content)
                docs.append(url_data)

            summary = summarize_chain.run(docs)
            response = quiz_chain(
                {
                    "text": summary,
                    "response_json": json.dumps(RESPONSE_JSON)
                }
            )
            print(response)


if __name__ == '__main__':
    main()
