import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities import WikipediaAPIWrapper

from utils import add_bg_from_local

from dotenv import load_dotenv
load_dotenv()

quizz_template = PromptTemplate(
    input_variables=['topic', 'wiki_search'],
    template="""
    Create a multiple choice quiz with 5 questions on the topic of {topic}. Leverage this wikipedia search: 
    {wiki_search} to create the questions. Each question should have 3 possible answers. Do not include the correct 
    answers in the quiz.
    Use the following format:
    1. <question>
      a. <option 1>
      b. <option 2>
      c. <option 3>
    """
)

llm = OpenAI(temperature=0.7, max_tokens=500)
wiki = WikipediaAPIWrapper()
script_chain = LLMChain(llm=llm, prompt=quizz_template, output_key="script", verbose=True)


def main():
    st.title("Quizz Generator")
    add_bg_from_local("images/abstract_1.jpg")
    prompt = st.text_input("Enter the topic you want to be tested on")
    if prompt:
        wiki_research = wiki.run(prompt)
        if wiki_research:
            quizz = script_chain.run(topic=prompt, wiki_search=wiki_research)
            st.markdown(quizz)
            with st.expander('Based on the Wikipedia Research:', expanded=False):
                st.info(wiki_research)
        else:
            st.write("This topic wasn't found on Wikipedia, please try another topic")


if __name__ == '__main__':
    main()
