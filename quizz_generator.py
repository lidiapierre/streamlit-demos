import json
import sys

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper

from utils import (
    hide_streamlit_header_footer,
    add_bg_from_local,
    get_text_splitter,
    get_llm, RESPONSE_JSON,
    get_table_data
)

quizz_template = PromptTemplate(
    input_variables=['text', 'response_json'],
    template="""
Text:{text}
Given the above text, create a quiz of 3 multiple choice questions. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \

RESPONSE_JSON:
{response_json}
"""
)

llm = get_llm()
wiki = WikipediaAPIWrapper()

map_prompt = """
Extract the key elements from:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = """
Extract the key elements from the text delimited by triple backquotes in about 10 sentences.
```{text}```
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

quiz_chain = LLMChain(llm=llm, prompt=quizz_template, output_key="quizz")

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
)

text_splitter = get_text_splitter()


def dirty_cleanup_response(str):
    i = 0
    while str[i] != '{':
        i += 1
    return str[i:]


def main():
    hide_streamlit_header_footer()
    st.title("Quizz Generator 📚")
    add_bg_from_local("images/abstract_1.jpg")

    with st.form("user_inputs"):
        topic = st.text_input("Enter the topic you want to be tested on")
        url = st.text_input("Optional: enter a link to the course content")
        button = st.form_submit_button("Create Quizz")

        if button:
            with st.spinner():
                docs = []
                if not topic:
                    sys.exit("Missing topic")
                wiki_search = wiki.load(topic)
                docs.extend(wiki_search)
                if url:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())

                chunks = text_splitter.split_documents(docs)

                summary = summary_chain.run(chunks)
                response = quiz_chain(
                    {
                        "text": summary,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )

                try:
                    quizz = response.get("quizz", None)
                    quizz = dirty_cleanup_response(quizz)
                    table_data = get_table_data(quizz)
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                    st.table(df)
                except Exception as e:
                    st.write(response)


if __name__ == '__main__':
    main()
