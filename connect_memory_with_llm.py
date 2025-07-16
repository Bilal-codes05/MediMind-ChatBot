from dotenv import load_dotenv
load_dotenv()

import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  

# Step 1: Load LLM from Groq
def load_llm():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192", 
        temperature=0.5,
        max_tokens=512
    )
    return llm

# Step 2: Prompt Template
CUSTOME_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Do not try to make up an answer.
Do not provide anything outside the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Step 3: Load FAISS vectorstore
DB_FAISS_PATH = "vectorestore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOME_PROMPT_TEMPLATE)
    }
)

# Step 5: Query the chain
user_query = input("Write Query here: ")
response = qa_chain.invoke({"query": user_query})

# Step 6: Output results
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"--- Document {i} ---")
    print(doc.page_content)
    print()
