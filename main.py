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
import json
import time
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "save_path" not in st.session_state:
        st.session_state.save_path = None

    # Load avatar image
    try:
        avatar = Image.open("avatar.png")
    except FileNotFoundError:
        st.error("Avatar image not found. Please ensure 'avatar.png' is in the same directory.")
        avatar = None

    # Display title with avatar
    col1, col2 = st.columns([1, 5]) # Adjust column widths as needed
    with col1:
        if avatar:
            st.image(avatar, width=50) # Adjust width as needed

    with col2:
        st.title("RAG z uploadowaniem PDF")


    uploaded_files = st.file_uploader("Wybierz pliki PDF", type="pdf", accept_multiple_files=True, help="Drag and drop PDF files here.", key="uploader")

    if uploaded_files:
        st.write("Uploaded files:")
        for file in uploaded_files:
            st.write(file.name)

        with st.spinner("Przetwarzanie plików..."):
            try:
                all_documents = []
                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    progress_bar = st.progress(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    os.remove(temp_file_path)
                    progress_bar.empty()

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(all_documents)
                embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
                db = FAISS.from_documents(texts, embeddings)
                st.session_state.db = db
                st.success("Pliki zostały przetworzone!")
            except Exception as e:
                st.error(f"Wystąpił błąd podczas przetwarzania plików: {e}")
                if "is not a valid file or url" in str(e):
                    st.error("Could not process one or more files. Please ensure they are valid PDFs.")
                else:
                    st.error("An unexpected error occurred while processing files. Please try again later.")

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

                    # Search bar for long answers
                    if len(response.text) > 500:
                        search_term = st.text_input("Search within answer:")
                        if search_term:
                            results = [i for i, line in enumerate(response.text.splitlines()) if search_term in line]
                            if results:
                                st.write(f"Results found on lines: {results}")
                            else:
                                st.write("No results found.")
                    st.write(response.text)

                    # Display conversation history
                    st.subheader("Historia konwersacji:")
                    for item in st.session_state.conversation:
                        st.write(f"**Pytanie:** {item['question']}")
                        st.write(f"**Odpowiedź:** {item['answer']}")

                    # Conversation history management
                    if st.button("Wyczyść historię"):
                        st.session_state.conversation = []
                    with st.expander("Zaawansowane"):
                        save_file = st.file_uploader("Wczytaj historię z pliku", type=["json"])
                        if save_file:
                            try:
                                st.session_state.conversation = json.load(save_file)
                                st.success("Historia została wczytana.")
                            except json.JSONDecodeError:
                                st.error("Nieprawidłowy format pliku JSON.")

                        save_path = st.text_input("Path to save history (optional):", st.session_state.save_path)
                        st.session_state.save_path = save_path
                        if st.button("Zapisz historię") and st.session_state.conversation:
                            try:
                                if save_path:
                                    with open(save_path, 'w') as f:
                                        json.dump(st.session_state.conversation, f, indent=4)
                                    st.success(f"Historia zapisana do {save_path}")
                                else:
                                    st.warning("Proszę podać ścieżkę zapisu.")
                            except Exception as e:
                                st.error(f"Błąd zapisu historii: {e}")
                                logging.error(f"Error saving history: {e}")
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas wyszukiwania odpowiedzi: {e}")
                    logging.error(f"Error during question answering: {e}")
                    if "There was a problem connecting to the Gemini API" in str(e):
                        st.error("There was a problem connecting to the Gemini API. Please check your API key and internet connection.")
                    else:
                        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
