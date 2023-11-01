import streamlit as st

def main():
    st.set_page_config(page_title="PDF Chat App", page_icon=":books:", layout="wide")

    st.header("PDF Chat App :books:")
    st.subheader("Upload and chat with your PDF file")

    st.text_input("Ask your question here:")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDF file here and click on 'Process'", type=["pdf"])
        st.button("Process")

if __name__ == "__main__":
    main()