# Ejecutar en terminal:
# python asistente_idiomas/rag_idioma.py

import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Sequence, Annotated, TypedDict, Literal

# --- 1. CONFIGURACIÓN INICIAL ---
def setup_environment():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable GEMINI_API_KEY no está configurada en .env")
    print("✅ Variables de entorno cargadas correctamente.")

# --- 2. CARGA DE DOCUMENTOS ---
def load_documents() -> list[Document]:
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("✅ Vectorstore creado correctamente con embeddings.")
    return vectorstore

# --- 4. HERRAMIENTAS ---
@tool
def off_topic_tool():
    """Se activa cuando el usuario pregunta algo que no sea vocabulario o idioma."""
    return "Lo siento, solo puedo ayudarte con vocabulario o traducciones de palabras."

def define_tools(vs):
    global vectorstore
    vectorstore = vs
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever,
        name="buscar_vocabulario",
        description="Busca palabras y sus traducciones en el documento de vocabulario inglés–español."
    )
    return [retriever_tool, off_topic_tool]

# --- 5. AGENTE ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state: AgentState, llm):
    system_prompt = """
    Eres 'Luna', un asistente virtual de idiomas. 
    Tu función es ayudar a traducir palabras y enseñar vocabulario en inglés y español.

    Instrucciones:
    - Usa la herramienta 'buscar_vocabulario' para responder preguntas sobre palabras o traducciones.
    - Si la pregunta no tiene que ver con vocabulario o idioma, usa 'off_topic_tool'.
    - Sé breve, clara y educativa.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"

def build_graph(llm_with_tools, tools_list):
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    graph.add_node("tools", ToolNode(tools_list))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    print("🧠 Grafo del asistente de idioma construido.")
    return graph.compile()

# --- 6. MAIN ---
if __name__ == "__main__":
    setup_environment()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    docs = load_documents()
    vectorstore = create_or_load_vectorstore(docs, embedding_model)
    tools = define_tools(vectorstore)
    llm_with_tools = llm.bind_tools(tools)
    rag_agent = build_graph(llm_with_tools, tools)

    print("\n\n🌍 Asistente de Idiomas 'Luna' listo para chatear 🌍")
    print("(Escribe 'salir' para terminar)\n")

    history = []
    while True:
        query = input("👤 Usuario: ")
        if query.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego! 🌟")
            break
        history.append(HumanMessage(content=query))
        result = rag_agent.invoke({"messages": history})
        history = result["messages"]
        print("🤖 Luna:", history[-1].content)
        
    # Agrega al final de tu rag_idioma.py
# Esto es solo para que main.py pueda importarlo directamente
def buscar_vocabulario(palabra: str):
    """
    Función auxiliar que usa el vectorstore para buscar vocabulario.
    Debes asegurarte de haber cargado el vectorstore antes de llamar.
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore no inicializado. Ejecuta define_tools primero.")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.get_relevant_documents(palabra)
    return [doc.page_content for doc in results]

vectorstore = None  # placeholder que se inicializa luego

