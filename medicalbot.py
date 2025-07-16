import streamlit as st 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorestore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.5,
        max_tokens=512
    )

def set_custom_prompt():
    template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Do not provide anything outside the given context.

    Context:
    {context}

    Question:
    {question}

    Start the answer directly. No small talk, please.
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_chain():
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=get_vectorstore().as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )

def main():
    st.set_page_config(page_title="MediMind Chatbot", page_icon="🩺")
    st.title("🧠 MediMind Chatbot")
    st.markdown("Ask me anything related to medical topics based on your uploaded documents.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a medical question...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                qa_chain = create_chain()
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]

                # Format source docs
                sources = response.get("source_documents", [])
                formatted_sources = ""
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get('source', f'Document {i}')
                    formatted_sources += f"\n\n**Source {i}:** `{source_name}`"

                final_answer = f"{result}\n\n---\n**Sources:**{formatted_sources}" if sources else result

                st.chat_message("assistant").markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Optional Reset Button
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()


