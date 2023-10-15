# LangChain & OpenAI Streamlit Apps

This repository contains a set of small Streamlit applications that leverage LangChain and OpenAI Large Language Models.

- **chat_with_own_data.py**: Chatbot agent to interact with your PDFs files and web contents.
- **quizz_generator.py**: Generate multiple choice quizz leveraging a Wikipedia search and online resources.
- **youtube_video_summary.py**: Summarize YouTube videos transcripts.

## How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/lidiapierre/streamlit-demo.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your personal keys

   ```bash
   # You can copy the example .env file
   cp example.env .env
   ```

4. Run each app using Streamlit:

   ```bash
   streamlit run chat_with_own_data.py
   streamlit run quizz_generator.py
   streamlit run youtube_video_summary.py
   ```
