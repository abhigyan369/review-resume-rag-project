📄 Personal Document Q&A Chatbot – Resume Advisor (RAG-Based)

A Retrieval-Augmented Generation (RAG) powered chatbot that allows users to upload personal PDF documents (especially resumes) and ask natural-language questions.
The system answers strictly from the uploaded document, making it ideal for resume analysis and interview preparation while minimizing hallucinations.

.

🚀 Features

📂 Upload your own PDF documents (resumes, notes, reports)

🔍 Ask natural-language questions about the document

🧠 Answers are grounded in document content only

❌ No hallucinations — refuses to answer if info is missing

💼 Acts as a Resume Advisor for interview preparation

🖥️ Simple and clean Streamlit UI

🎯 Why This Project?

Most beginner chatbot projects rely solely on prompts and often hallucinate.
This project demonstrates real-world LLM engineering by using Retrieval-Augmented Generation (RAG) to ensure factual accuracy.

Unique Angle:
Instead of a generic chatbot, this system behaves like a resume-aware interview assistant, making it practically useful for job seekers.

Example Questions You Can Ask

What skills are listed in my resume?

What projects have I worked on?

Summarize my experience in simple words

What technologies do I have experience with?

What should I highlight in a technical interview?

If the answer is not found in the document, the chatbot responds:

“This information is not present in the uploaded document.”

🏗️ Architecture Overview (RAG Pipeline)

1.PDF Upload & Text Extraction

2.Text Chunking with Overlap

3.Embedding Generation

4.Vector Storage (FAISS / Chroma)

5.Query Embedding & Similarity Search

6.Context-Aware Answer Generation

Tech Stack

| Component       | Technology              |
| --------------- | ----------------------- |
| Language        | Python                  |
| RAG Framework   | LangChain               |
| Vector Database | FAISS                   |
| LLM             |HuggingFace API          |
| Frontend        | Streamlit               |

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/personal-doc-chatbot.git
cd personal-doc-chatbot
2️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py
Open your browser and upload a PDF to start chatting.

📈 Future Enhancements

Support for multiple document uploads

Chat history & conversational memory

Highlighted source references in answers

Resume scoring and improvement suggestions

Deployment using Docker / Cloud

🤝 Contributing

Contributions, suggestions, and improvements are welcome.
Feel free to fork the repository and submit a pull request.
