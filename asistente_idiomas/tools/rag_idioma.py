# asistente_idiomas/rag_idioma.py
# Este módulo se encarga de cargar el conocimiento (vocabulario, frases, gramática)
# y preparar el RAG con embeddings y búsqueda semántica.

import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    """
    Carga todos los archivos .txt desde la carpeta 'knowledge'.
    Cada archivo se convierte en un Document independiente con metadata.
    """
    knowledge_dir = "asistente_idiomas/knowledge"
    if not os.path.exists(knowledge_dir):
        raise FileNotFoundError(f"No se encontró la carpeta: {knowledge_dir}")

    documents = []
    total_chars = 0

    for filename in os.listdir(knowledge_dir):
        if filename.endswith(".txt"):
            path = os.path.join(knowledge_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                    total_chars += len(text)
                    print(f"📘 Documento cargado: {filename} ({len(text)} caracteres)")
                else:
                    print(f"⚠️ Archivo vacío omitido: {filename}")

    if not documents:
        raise ValueError("No se encontraron archivos .txt con contenido en la carpeta 'knowledge'.")

    print(f"📚 Total de documentos cargados: {len(documents)} ({total_chars} caracteres en total)")
    return documents


# --- 3. CREACIÓN DEL VECTORSTORE ---
def create_or_load_vectorstore(documents, embedding_model):
    """Crea el vectorstore de Chroma usando embeddings del modelo de Google."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print(f"✅ Vectorstore creado correctamente con {len(splits)} fragmentos.")
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
    print("🌐 RAG de idioma inicializado correctamente con todos los archivos .txt.")


def buscar_vocabulario(palabra: str):
    """
    Busca una palabra o frase en el vectorstore y devuelve los fragmentos más relevantes.
    Asegúrate de haber ejecutado 'inicializar_rag()' antes de llamar a esta función.
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore no inicializado. Ejecuta inicializar_rag() primero.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(palabra)

    print(f"🔍 Búsqueda realizada: '{palabra}' → {len(results)} resultados relevantes.")
    return [f"[{doc.metadata.get('source', 'sin fuente')}] {doc.page_content}" for doc in results]


# --- 5. RESPUESTA PARA TEMAS FUERA DE CONTEXTO ---
def off_topic_tool():
    """Responde cuando el usuario pregunta algo fuera del contexto de aprendizaje de idiomas."""
    return "❗ Lo siento, solo puedo ayudarte con temas de idiomas. Por favor, haz preguntas relacionadas con vocabulario, frases o traducciones."
