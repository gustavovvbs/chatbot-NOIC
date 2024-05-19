from dotenv import load_dotenv
import os
import numpy as np
from langchain.vectorstores import FAISS 
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

vec_db = FAISS.load_local("noic_index", embeddings, allow_dangerous_deserialization=True)

retriever = vec_db.as_retriever()

def junta_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def get_prompt(query):
    return f"""Responda a query utilizando o contexto disponibilizado:

    Query: {query}

    Contexto:{junta_docs(retriever.invoke(query))}"""

