from dotenv import load_dotenv
import os
import numpy as np
from langchain.vectorstores import Chroma, FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader 
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncHtmlLoader
from langsmith import Client 
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import traceable
from pdr import pdr_build


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "TESTE"

client = Client()


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)



def junta_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])
@traceable
def get_prompt(query, contexto):
    return f"""Responda a query utilizando o contexto disponibilizado:

    Query: {query}

    Contexto:{junta_docs(contexto)}"""


