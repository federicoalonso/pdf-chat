import math
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores.pgvector import PGVector
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

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

def get_top3_similar_docs(query_embedding, cur, conn):
    embedding_array = np.array(query_embedding)
    register_vector(conn)
    # Get the top 3 most similar documents using the KNN <=> operator
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

def main():
    # set debbuging to true
    st.set_option('deprecation.showfileUploaderEncoding', False)
    load_dotenv()
    conn = configure_db()
    st.set_page_config(page_title="PDF Chat App", page_icon=":books:", layout="wide")

    st.header("PDF Chat App :books:")
    st.subheader("Upload and chat with your PDF file")

    st.text_input("Ask your question here:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doct = st.file_uploader("Upload your PDF file here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_doct)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                cur = conn.cursor()
                insert_db(vectorstore, text_chunks, cur, conn)
                st.write(vectorstore)
                st.success("Done!")

if __name__ == "__main__":
    main()