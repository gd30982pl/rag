import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import traceback
import tempfile
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    st.title("RAG z uploadowaniem PDF")

    uploaded_files = st.file_uploader("Wybierz pliki PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Przetwarzanie plików..."):
            try:
                all_documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    os.remove(temp_file_path)

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(all_documents)
                embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
                db = FAISS.from_documents(texts, embeddings)
                st.session_state.db = db
                st.success("Pliki zostały przetworzone!")
            except Exception as e:
                st.error(f"Wystąpił błąd podczas przetwarzania plików: {e}")
                st.error(traceback.format_exc())


    query = st.text_input("Zadaj pytanie:")

    if st.button("Zadaj pytanie") and st.session_state.db:
        with st.spinner("Szukanie odpowiedzi..."):
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Klucz API Gemini nie został znaleziony w pliku .env. Utwórz plik .env z kluczem w zmiennej GEMINI_API_KEY.")
                    return
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                retriever = st.session_state.db.as_retriever()
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"Answer the following question using the provided context:\n\nQuestion: {query}\n\nContext: {context}"

                response = model.generate_content(prompt)
                st.session_state.conversation.append({"question": query, "answer": response.text})
                st.write(response.text)

                # Display conversation history
                st.subheader("Historia konwersacji:")
                for item in st.session_state.conversation:
                    st.write(f"**Pytanie:** {item['question']}")
                    st.write(f"**Odpowiedź:** {item['answer']}")

            except Exception as e:
                st.error(f"Wystąpił błąd podczas wyszukiwania odpowiedzi: {e}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
