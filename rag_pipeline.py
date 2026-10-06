import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize HuggingFace Embeddings and Groq LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama3-8b-8192")

def process_document(file_path):
    """Reads a PDF, chunks the text, and stores it in ChromaDB."""
    # 1. Load the PDF document
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create and persist the Vector Database
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vectorstore

def answer_question(vectorstore, question):
    """Retrieves relevant context and generates an answer using LLM."""
    retriever = vectorstore.as_retriever()
    
    # Define the system prompt for the AI
    system_prompt = (
        "You are an intelligent AI research assistant. Use the following context "
        "to answer the user's question accurately. If the answer is not contained "
        "in the context, clearly state that you do not know based on the provided document.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Setup the RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Execute the query and return the answer
    response = rag_chain.invoke({"input": question})
    return response["answer"]