import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_rag_chain(vector_store):
    """
    Creates a RetrievalQA chain using the provided vector store and HuggingFace LLM.
    """
    
    # 1. LLM Setup
    # Ensure HUGGINGFACEHUB_API_TOKEN is set in environment
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize Endpoint with explicit task="text-generation" might fail if API rejects it.
    # However, ChatHuggingFace needs an LLM. 
    # Attempting to use the endpoint as is. If 'task' defaults to text-generation, we might need to handle it.
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation", # We'll try text-generation first, as ChatHuggingFace often expects a base generator
        temperature=0.1,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    # Wrap in ChatHuggingFace to handle Intruct/Chat models properly
    chat_model = ChatHuggingFace(llm=llm)

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
        llm=chat_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": chat_prompt},
        return_source_documents=True
    )
    
    return qa_chain
