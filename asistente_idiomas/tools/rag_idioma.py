# asistente_idiomas/tools/rag_idioma.py
import os
import json
import hashlib
import shutil
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

FINGERPRINT_FILE = "data/chroma_db/_docs_fingerprint.json"
vectorstore = None  # variable global

def setup_environment():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable GEMINI_API_KEY no está configurada en el archivo .env")
    print("✅ Variables de entorno cargadas correctamente.")

def load_documents() -> list[Document]:
    knowledge_dir = "asistente_idiomas/knowledge"
    if not os.path.exists(knowledge_dir):
        raise FileNotFoundError(f"No se encontró la carpeta: {knowledge_dir}")

    documents = []
    total_chars = 0

    for filename in sorted(os.listdir(knowledge_dir)):
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

# ---------- fingerprint helpers ----------
def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _compute_docs_fingerprint(knowledge_dir: str) -> dict:
    files = sorted([f for f in os.listdir(knowledge_dir) if f.endswith(".txt")])
    fp = {}
    for fn in files:
        path = os.path.join(knowledge_dir, fn)
        fp[fn] = _file_hash(path)
    return fp

def _read_saved_fingerprint() -> dict | None:
    if os.path.exists(FINGERPRINT_FILE):
        try:
            with open(FINGERPRINT_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

def _save_fingerprint(fp: dict):
    os.makedirs(os.path.dirname(FINGERPRINT_FILE), exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as fh:
        json.dump(fp, fh, ensure_ascii=False, indent=2)

# ---------- vectorstore creation / load ----------
def create_or_load_vectorstore(docs, embedding_model, force_rebuild: bool = False):
    # splitter apropiado para glosarios
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n- ", "\n", ".", " "],
        chunk_size=150,
        chunk_overlap=20
    )
    splits = splitter.split_documents(docs)

    persist_dir = "data/chroma_db"
    knowledge_dir = "asistente_idiomas/knowledge"
    current_fp = _compute_docs_fingerprint(knowledge_dir)
    saved_fp = _read_saved_fingerprint()

    # decide si reconstruir
    should_rebuild = force_rebuild
    if not should_rebuild:
        if saved_fp is None:
            print("⚠️ No se encontró fingerprint previo: se reconstruirá vectorstore.")
            should_rebuild = True
        elif saved_fp != current_fp:
            print("⚠️ Cambios detectados en 'knowledge/' -> se reconstruirá vectorstore.")
            should_rebuild = True
        else:
            should_rebuild = False

    if not should_rebuild and os.path.exists(persist_dir):
        print("💾 Cargando vectorstore existente desde disco...")
        # algunos wrappers esperan embedding_function, otros embedding=modelo; esto funciona en la mayoría
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        return vectorstore

    # Si llegamos acá, creamos el vectorstore nuevo
    if os.path.exists(persist_dir):
        # eliminar DB previa para evitar mezcla con embeddings antiguos
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print("⚠️ No se pudo eliminar el persist_dir previo:", e)

    print("🧠 Creando vectorstore nuevo (esto puede tardar un poco)...")
    # from_documents suele crear y persistir según la versión
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    # Algunas versiones antiguas/objetos tienen `.persist()`, otras no.
    if hasattr(vectorstore, "persist"):
        try:
            vectorstore.persist()
        except Exception as e:
            print("⚠️ vectorstore.persist() falló:", e)
    else:
        print("ℹ️ Nota: objeto Chroma no tiene método persist(); asumo que from_documents ya persistió la DB.")

    # guardamos fingerprint
    _save_fingerprint(current_fp)
    print("✅ Vectorstore creado y fingerprint guardado.")
    return vectorstore

def inicializar_rag(force_rebuild: bool = False):
    global vectorstore
    setup_environment()

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    docs = load_documents()
    vectorstore = create_or_load_vectorstore(docs, embedding_model, force_rebuild=force_rebuild)
    print("🌐 RAG de idioma inicializado correctamente con todos los archivos .txt.")

def buscar_vocabulario(palabra: str):
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore no inicializado. Ejecuta inicializar_rag() primero.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # warning: get_relevant_documents puede estar deprecado en tu versión; lo usamos por compatibilidad
    try:
        results = retriever.get_relevant_documents(palabra)
    except Exception:
        # fallback a invoke si la versión nueva lo exige
        results = retriever.invoke({"input": palabra}).get("output", [])

    print(f"🔍 Búsqueda realizada: '{palabra}' → {len(results)} resultados relevantes.")
    for i, doc in enumerate(results):
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source", "sin fuente")
        preview = getattr(doc, "page_content", str(doc))[:200]
        print(f"  - Resultado {i}: fuente={source} preview={preview!r}")
    return [f"[{getattr(doc, 'metadata', {}).get('source', 'sin fuente')}] {getattr(doc, 'page_content', str(doc))}" for doc in results]

def off_topic_tool():
    return "❗ Lo siento, solo puedo ayudarte con temas de idiomas. Por favor, haz preguntas relacionadas con vocabulario, frases o traducciones."
