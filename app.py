import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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

def main():
    load_dotenv()
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
                st.write(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()