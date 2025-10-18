# asistente_idiomas/rag_idioma.py
# Este módulo solo se usa como soporte del asistente principal (no se ejecuta directamente)

import os
from dotenv import load_dotenv
from langchain.schema import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- 1. CONFIGURACIÓN INICIAL ---
def setup_environment():
    """Carga las variables de entorno necesarias."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable GEMINI_API_KEY no está configurada en el archivo .env")
    print("✅ Variables de entorno cargadas correctamente.")


# --- 2. CARGA DE DOCUMENTOS ---
def load_documents() -> list[Document]:
    """Carga el documento de vocabulario desde la carpeta knowledge."""
    vocab_path = "asistente_idiomas/knowledge/vocabulario.txt"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"No se encontró el archivo: {vocab_path}")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_text = f.read()

    docs = [Document(page_content=vocab_text, metadata={"source": "vocabulario.txt"})]
    print(f"📘 Documento cargado: vocabulario.txt ({len(vocab_text)} caracteres)")
    return docs


# --- 3. CREACIÓN DEL VECTORSTORE ---
def create_or_load_vectorstore(documents, embedding_model):
    """Crea el vectorstore de Chroma usando embeddings del modelo de Google."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("✅ Vectorstore creado correctamente con embeddings.")
    return vectorstore


# --- 4. INICIALIZACIÓN Y FUNCIÓN AUXILIAR ---
vectorstore = None  # Variable global inicializada desde main.py


def inicializar_rag():
    """Inicializa todo el entorno del RAG (entorno, embeddings, vectorstore)."""
    global vectorstore
    setup_environment()

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    docs = load_documents()
    vectorstore = create_or_load_vectorstore(docs, embedding_model)
    print("🌐 RAG de idioma inicializado correctamente.")


def buscar_vocabulario(palabra: str):
    """
    Busca una palabra o frase en el vectorstore y devuelve los fragmentos más relevantes.
    Asegúrate de haber ejecutado 'inicializar_rag()' antes de llamar a esta función.
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore no inicializado. Ejecuta inicializar_rag() primero.")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.get_relevant_documents(palabra)
    return [doc.page_content for doc in results]

# en rag_idioma.py o en main.py
def off_topic_tool():
    """
    Responde cuando el usuario pregunta algo fuera del contexto de aprendizaje de idiomas.
    Solo devuelve un mensaje genérico.
    """
    return "❗ Lo siento, solo puedo ayudarte con temas de idiomas. Por favor, haz preguntas relacionadas con vocabulario, frases o traducciones."

