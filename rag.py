from dotenv import load_dotenv
import os
import numpy as np
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader 
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncHtmlLoader
import pickle



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

# def ainvokef(prompt):
#     with open('retriever.pkl', 'rb') as input:
#         retriever = pickle.load(input)

#     return retriever.invoke(prompt)

def stdinvoke(prompt):
    vec_db = Chroma(
        collection_name = 'split_parents', embedding_function = embeddings, persist_directory='./chroma_db'
    )
    docs = vec_db.similarity_search(prompt, k=5)
    print(docs)

retrieves = ainvokef('matematica basicaw')
print(retrieves)

def junta_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def get_prompt(query):
    return f"""Responda a query utilizando o contexto disponibilizado:

    Query: {query}

    Contexto:{junta_docs(ainvoke(query))}"""


