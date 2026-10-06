import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables (API keys)
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

# Sidebar for file uploading
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")
        # We will connect the PDF reading logic here later

# Main question input area
st.subheader("Ask a Question")
user_question = st.text_input("What would you like to know about this document?")

# Handle user query
if user_question:
    if uploaded_file is not None:
        with st.spinner("Searching for answers..."):
            # AI response logic will be added here
            st.info("AI is analyzing the document...")
    else:
        st.warning("Please upload a PDF document first.")