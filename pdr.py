from dotenv import load_dotenv
import os
import numpy as np
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore, InMemoryStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader 
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncHtmlLoader
from langchain.storage import LocalFileStore
import pickle
from langsmith import Client, traceable

langclient = Client(api_key=os.getenv("LANGCHAIN_API_KEY"), api_url="https://api.smith.langchain.com")




load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

@traceable
def pdr_build():
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

    fs = LocalFileStore('./chroma_db')
    store = create_kv_docstore(fs)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=900)
    
    vectorstore = Chroma(embedding_function = embeddings, collection_name = 'split_parents')
    retriever = ParentDocumentRetriever(
        vectorstore = vectorstore,
        docstore = store,
        child_splitter = child_splitter,
        parent_splitter = parent_splitter,
    )
    retriever.add_documents(todes)
    
    return retriever

def save_object(obj, filename):
    with open(filename, 'wb') as output: 
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


