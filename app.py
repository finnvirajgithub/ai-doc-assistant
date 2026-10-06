import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_pipeline import process_document, answer_question

# Load environment variables
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide"
)

# Main Header
st.title("📄 AI Document & Research Assistant")
st.markdown("Upload any PDF research paper or document and ask questions instantly.")

# Initialize session state to store the vector database
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar for file uploading
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and st.session_state.vectorstore is None:
        with st.spinner("Processing PDF... Please wait."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            # Process the document and store in ChromaDB
            st.session_state.vectorstore = process_document(temp_file_path)
            
            # Remove the temporary file
            os.remove(temp_file_path)
            
        st.success("PDF processed and ready for questions!")

# Main question input area
st.subheader("Ask a Question")
user_question = st.text_input("What would you like to know about this document?")

# Handle user query
if user_question:
    if st.session_state.vectorstore is not None:
        with st.spinner("Searching for answers..."):
            # Get the answer from the RAG pipeline
            answer = answer_question(st.session_state.vectorstore, user_question)
            
            # Display the answer
            st.markdown("### Answer:")
            st.info(answer)
    else:
        st.warning("Please upload a PDF document first.")