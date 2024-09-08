import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat APP')
    st.markdown('''
## About 
This app is an LLM-powered chatbot built using:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://platform.openai.com/docs/models) LLM model
''')
    add_vertical_space(5)  # Adds vertical space
    st.write('Made with ❤️ by [Maria Engineer]')

def main():
    st.header("Chat with PDF")
    load_dotenv()
    
    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        embeddings = OpenAIEmbeddings()

        # Create FAISS vector store from text chunks
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Save the vector store to a pickle file
        store_name = pdf.name[:-4]  # Remove the ".pdf" extension
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        
        st.write(f"Vector store saved as {store_name}.pkl")

if __name__ == '__main__':
    main()
