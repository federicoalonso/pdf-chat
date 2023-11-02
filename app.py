import math
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from langchain.llms.openai import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory,
    CombinedMemory,
    ConversationSummaryBufferMemory
)
from langchain.memory.chat_message_histories import RedisChatMessageHistory

from htmlTemplate import css, bot_template, user_template

def configure_db():
    conn_string = os.getenv("CONNECTION_STRING")
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    table_create_command = """
    CREATE TABLE IF NOT EXISTS embeddings (
                id bigserial primary key, 
                content text,
                embedding vector(1536)
                );
                """

    cur.execute(table_create_command)
    cur.close()
    conn.commit()
    return conn

def insert_db(embeddings, texts, cur, conn):
    execute_values(cur, "INSERT INTO embeddings (content, embedding) VALUES %s", [(text, embedding) for embedding, text in zip(embeddings, texts)])
    cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
    num_records = cur.fetchone()[0]
    print("Number of vector records in table: ", num_records,"\n")
    num_lists = num_records / 1000
    if num_lists < 10:
        num_lists = 10
    if num_records > 1000000:
        num_lists = math.sqrt(num_records)

    cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});')
    conn.commit()
    cur.close()

def get_top3_similar_docs(query_embedding):
    embedding_array = np.array(query_embedding)
    register_vector(st.session_state.conn)
    # Get the top 3 most similar documents using the KNN <=> operator
    cur = st.session_state.conn.cursor()
    cur.execute("SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT 3", (embedding_array,))
    top3_docs = cur.fetchall()
    return top3_docs

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    document_vectors = embeddings.embed_documents([t for t in text_chunks])
    return document_vectors

def get_completion_from_messages(prompt, memory, llm):
    
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory, 
        prompt=prompt
    )

    response = conversation.predict()

    st.write(response)

    return response

# Helper function: get embeddings for a text
def get_embeddings(text):
    embeddings = OpenAIEmbeddings()
    document_vectors = embeddings.embed_documents([text])
    return document_vectors[0]

def process_input_with_retrieval(user_input):
    email = "fede"
    message_history = RedisChatMessageHistory(
        url="redis://localhost:6379/0", ttl=1200, session_id=email
    )

    summary_history = RedisChatMessageHistory(
        url="redis://localhost:6379/0", ttl=600, session_id=email + "_summary"
    )

    llm = OpenAIChat(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1500,
    )

    conv_memory = ConversationBufferWindowMemory(
        memory_key="chat_history_lines", chat_memory=message_history, input_key="input", k=5
    )

    #Step 1: Get documents related to the user input from database
    related_docs = get_top3_similar_docs(get_embeddings(user_input))

    assistant_memory = f"Información relevante para asistirte: \n {related_docs[0][0]} \n {related_docs[1][0]} {related_docs[2][0]}"

    summary_memory = ConversationSummaryBufferMemory(llm=llm, input_key="input", chat_memory=summary_history, max_token_limit=1500)

    _DEFAULT_TEMPLATE = """Eres un bot que constesta las preguntas de un humano, brindando mayor importancia a la información obtenida de los documentos relevantes
    Documentos relevantes:
    """ + assistant_memory + """
    Summary of conversation:
    {history}
    Current conversation:
    {chat_history_lines}
    Humano: {input}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input", "chat_history_lines"],
        template=_DEFAULT_TEMPLATE,
    )

    memory = CombinedMemory(memories=[conv_memory, summary_memory])

    conversation = ConversationChain(
        llm=llm, 
        verbose=False, 
        memory=memory, 
        prompt=PROMPT
    )

    response = conversation.predict(input=user_input)

    return response


def handle_user_question(user_question):
    response = process_input_with_retrieval(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": response})

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    load_dotenv()

    if "conn" not in st.session_state:
        st.session_state.conn = configure_db()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="PDF Chat App", page_icon=":books:", layout="wide")

    st.header("PDF Chat App :books:")
    st.subheader("Upload and chat with your PDF file")

    st.write(css, unsafe_allow_html=True)

    user_question = st.text_input("Ask your question here:")
    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doct = st.file_uploader("Upload your PDF file here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_doct)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                cur = st.session_state.conn.cursor()
                insert_db(vectorstore, text_chunks, cur, st.session_state.conn)

if __name__ == "__main__":
    main()