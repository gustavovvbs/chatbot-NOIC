from dotenv import load_dotenv
import os
import streamlit as st
import numpy as np
import openai
from rag import junta_docs
import langsmith
from langsmith.wrappers import wrap_openai
from pdr import pdr_build
from langsmith import traceable


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "TESTE"

client = langsmith.Client(api_url="https://api.smith.langchain.com", api_key=os.getenv("LANGCHAIN_API_KEY"))

st.title('Assistente olimpico')
openai.api_key = os.getenv("OPENAI_API_KEY")

openaiclient = wrap_openai(openai.Client())

if "parentretriever" not in st.session_state:
    st.session_state.parentretriever = pdr_build()

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"


@traceable
def get_prompt(query):
    return f"""Responda a query utilizando o contexto disponibilizado:

    Query: {query}
    
    Contexto:{junta_docs(st.session_state.parentretriever.invoke(query))}"""

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({'role': 'system', 'content': 'Você é o Ualype, chatbot do site NOIC. Você é um funcionário do NOIC, e deve responder as perguntas dos usuários. Seja educado e prestativo. Você deve usar o contexto disponível para responder as perguntas dos usuários. Se não souber, não minta, e nem faça menções sobre consulta em documentos. Caso não saiba, mande o usuário entrar no grupo de whatsapp ou discord do NOIC.Recomendações de Material de Estudo: Para as primeiras fases da seletiva e a fase de Barra:  Recomende o livro "Astronomia Olímpica" do NOIC. Para a fase de Vinhedo: Recomende a apostila Magna.'})

    st.session_state.messages.append({'role': 'assistant', 'content': 'Olá!, sou o Ualype, a inteligência artificial do noic que vai te ajudar a ser medalhista em olimpíadas! Como posso te ajudar hoje?'})


for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
                st.markdown(message['content'])

prompt = st.chat_input("Digite sua pergunta aqui...")

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    augmentedprompt = get_prompt(prompt)

    messages_copy = st.session_state.messages.copy()
    messages_copy.append({'role': 'user', 'content': augmentedprompt})

    with st.chat_message('assistant'):
        stream = openaiclient.chat.completions.create(
            model = st.session_state['openai_model'],
            messages = [
                {'role': m['role'], 'content': m['content']} for m in messages_copy
            ],
            stream = True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({'role': 'assistant', 'content': response})