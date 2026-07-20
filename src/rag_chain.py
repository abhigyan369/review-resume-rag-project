import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_rag_chain(vector_store):
    """
    Creates a RetrievalQA chain using the provided vector store and Groq LLM.
    """
    
    # 1. LLM Setup
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing or empty. Please set it in your .env file.")

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=groq_api_key
    )

    # 2. Prompt Template (Chat Format)
    system_template = """You are a helpful assistant for analyzing resumes. Use the following pieces of context to answer the question at the end.
    
    If the answer is not present in the context, say "This information is not present in the uploaded document."
    Do not try to make up an answer.
    
    Context:
    {context}"""
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # 3. Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": chat_prompt},
        return_source_documents=True
    )
    
    return qa_chain
