import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import add_bg_from_local

from dotenv import load_dotenv
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
llm = OpenAI(temperature=0.6)
chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=False)


def main():
    add_bg_from_local("images/abstract_5.jpg")
    st.header("Youtube video summarizer")
    url = st.text_input("Youtube video URL", placeholder="https://www.youtube.com/watch?xxxxxx")

    if url:
        load_video = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=True)
        result = load_video.load()

        texts = text_splitter.split_documents(result)
        result = chain.run(texts)

        st.markdown("**Summary**")
        st.write(result)


if __name__ == '__main__':
    main()
