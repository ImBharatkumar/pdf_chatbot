
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sys

# Load the .env file
load_dotenv()
# Get the open_api_key from the .env file
open_api_key = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAIEmbeddings object
openai_embeddings = OpenAIEmbeddings(api_key=open_api_key)
print(open_api_key)