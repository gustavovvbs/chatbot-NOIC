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



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

def ainvoke(prompt):
    urls = ["https://noic.com.br/olimpiadas/", "https://noic.com.br/olimpiadas/astronomia/", "https://noic.com.br/astronomia/guia/", "https://noic.com.br/astronomia/guia/oba/", "https://noic.com.br/astronomia/guia/vinhedo/", "https://noic.com.br/astronomia/guia/antigos-e-olaa/"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    loaderonline = PyPDFLoader('./Guia_NOIC_Seletivas_Online.pdf')
    loaderbarra = PyPDFLoader('./guiabarra.pdf')

    online = loaderonline.load()
    barra = loaderbarra.load()

    barraonline = barra + online

    todes = barraonline + docs_transformed
    vec_db = Chroma(
        collection_name = 'split_parents', embedding_function = embeddings
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore = vec_db,
        docstore = store,
        child_splitter = child_splitter,
        parent_splitter = parent_splitter,
    )
    retriever.add_documents(todes)

    return retriever.invoke(prompt)

print(ainvoke('seletiva'))

def junta_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def get_prompt(query):
    return f"""Responda a query utilizando o contexto disponibilizado:

    Query: {query}

    Contexto:{junta_docs(ainvoke(query))}"""

