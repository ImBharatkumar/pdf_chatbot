from src.logger import logging
from src.exception import CustomException
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableMap

template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
model = model = OllamaLLM(model="llama3.2", extra_fields_behavior="allow")  # Adjust as per documentation
chain = RunnableMap({
    "context": lambda x: "\n".join([doc.page_content for doc in x["input_documents"]]),
    "question": lambda x: x["question"]
}) | model


# Load the .env file
load_dotenv()
# Get the _api_key from the .env file
jina_api_key = os.getenv("JINA_API_KEY")



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
        if not uploaded:
            st.warning("Please upload a PDF file first.")
            return

        reader = PdfReader(uploaded)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docsearch = FAISS.from_texts(texts, embeddings)

        # Load the QA chain using the model
        chain = load_qa_chain(model, chain_type="stuff")
        docs = docsearch.similarity_search(query)

        result = chain.invoke({"input_documents": docs, "question": query})

        # Extract and return only the answer text
        if isinstance(result, dict) and "output_text" in result:
            return result["output_text"]
        else:
            return result


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

