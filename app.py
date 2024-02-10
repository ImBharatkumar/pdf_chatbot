from src.logger import logging
from src.exception import CustomException
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sys
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the .env file
load_dotenv()
# Get the open_api_key from the .env file
open_api_key = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAIEmbeddings object
openai_embeddings = OpenAIEmbeddings(api_key=open_api_key)


try:
    logging.info("session started")
    st.header('Document chat bot')
    # session_state is used to store or display chat history
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # location of the pdf file/files.
    uploaded = st.file_uploader('choose a file')

    # read data from the file and put them into a variable called raw_text
    def get_ans(query):
        reader = PdfReader(uploaded)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        print(text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


        # FAISS is a library for efficient similarity search and clustering of dense vectors
        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = docsearch.similarity_search(query)
        return chain.run(input_documents=docs, question=query)

    def clear_text_input():
        global input_text
        input_text=""

    def get_text():
        global input_text
        input_text= st.text_input('Ask a question',key='input',on_change=clear_text_input)
        return input_text

    def clear_history():
        st.session_state['generated'] = []
        st.session_state['past'] = []

    if uploaded:
        user_input= get_text()
        if st.button('post'):
            output=get_ans(user_input)
            # storing the output
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.button('Clear History'):
        clear_history()

    logging.info('message generated')
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        message(st.session_state['past'][i],is_user=True,key=f"user_message_{i}")
        message(st.session_state['generated'][i],key=str(i))
except Exception as e:
    raise CustomException(error_message="An error occurred", error_detail=e)