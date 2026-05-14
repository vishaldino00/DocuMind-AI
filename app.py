
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("📄 DocuMind AI")
st.subheader("Chat with your PDF")

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    query = st.text_input("Ask a question")

    if query:

        results = vector_store.similarity_search(query, k=3)

        context = "\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        st.write("### 🤖 AI Answer")

        st.write(response.choices[0].message.content)
