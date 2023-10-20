import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain

from utils import add_bg_from_local, get_llm, get_text_splitter, hide_streamlit_header_footer


def main():
    hide_streamlit_header_footer()
    add_bg_from_local("images/abstract_5.jpg")
    st.header("Youtube video summarizer")
    url = st.text_input("Youtube video URL", placeholder="https://www.youtube.com/watch?xxxxxx")

    llm = get_llm()
    text_splitter = get_text_splitter()
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=False)

    if url:
        with st.spinner('Wait for it...'):
            load_video = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=True)
            result = load_video.load()
            texts = text_splitter.split_documents(result)
            result = chain.run(texts)

        st.markdown("**Summary**")
        st.write(result)


if __name__ == '__main__':
    main()
