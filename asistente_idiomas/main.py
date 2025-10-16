# asistente_idiomas/main.py
import os
from typing import Sequence, Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# Tus agentes
from asistente_idiomas.agentes.tutor import Tutor
from asistente_idiomas.agentes.registrador import Registrador

# Herramientas
from asistente_idiomas.tools.rag_idioma import buscar_vocabulario
from asistente_idiomas.tools.notion_tool import guardar_en_notion

# --- 1️⃣ Configuración de entorno ---
def setup_environment():
    """Carga variables desde el archivo .env y valida la API key."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("⚠️ Falta la variable GEMINI_API_KEY en tu archivo .env")
    print("✅ Variables de entorno cargadas correctamente.")


# --- 2️⃣ Definición de herramientas ---
@tool
def herramienta_buscar_vocabulario(palabra: str):
    """Busca definiciones o traducciones de palabras en la base RAG de idiomas."""
    return buscar_vocabulario(palabra)

@tool
def herramienta_guardar_en_notion(texto: str):
    """Guarda un resumen o nota en Notion."""
    return guardar_en_notion(texto)

@tool
def herramienta_off_topic():
    """Responde cuando el usuario pregunta algo fuera del contexto educativo."""
    return detectar_off_topic()

def definir_herramientas():
    print("🛠️ Herramientas cargadas: vocabulario, guardar_en_notion, off_topic.")
    return [
        herramienta_buscar_vocabulario,
        herramienta_guardar_en_notion,
        herramienta_off_topic,
    ]


# --- 3️⃣ Estado compartido ---
class AgentState(TypedDict):
    """Estado compartido entre nodos del grafo."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    registrar: bool  # indica si se debe registrar


# --- 4️⃣ Nodos del grafo ---
def nodo_tutora(state: AgentState, llm, tutor: Tutor):
    """
    Nodo principal: llama al Tutor y decide si usar herramientas.
    Devuelve estado con mensajes y registrar.
    """
    system_prompt = """
    Eres "Luna", tutora de idiomas amable y paciente 🌙.
    Tu misión es ayudar al usuario a mejorar vocabulario y fluidez en idiomas.
    
    Instrucciones:
    1. Saluda y mantén actitud empática y didáctica.
    2. Usa herramientas RAG para buscar palabras o frases.
    3. Usa Notion si el usuario pide guardar algo.
    4. Si el mensaje no tiene relación con aprendizaje, usa off-topic.
    5. Responde de manera clara y educativa.
    """
    
    mensajes = [SystemMessage(content=system_prompt)] + state["messages"]
    # Usamos tutor para procesar la última pregunta del usuario
    if state["messages"]:
        ultima = state["messages"][-1]
        if isinstance(ultima, HumanMessage):
            resultado = tutor.responder(ultima.content)
            state["registrar"] = resultado.get("registrar", False)
            # Añadimos respuesta de tutor como HumanMessage para contexto
            tutor_message = HumanMessage(content=resultado.get("respuesta", ""))
            state["messages"].append(tutor_message)
    
    # Invocamos LLM con el historial completo
    respuesta = llm.invoke(state["messages"])
    state["messages"].append(respuesta)
    
    return state


def nodo_registrador(state: AgentState, registrador: Registrador):
    """Registra aprendizajes si corresponde."""
    if state.get("registrar"):
        ultima = state["messages"][-1].content if state["messages"] else ""
        registrador.registrar(f"Nuevo aprendizaje: {ultima}")
        state["registrar"] = False  # reseteamos
    return state


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determina si hay que ejecutar herramientas o terminar."""
    if state["messages"] and state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"


def construir_grafo(llm_con_herramientas, lista_herramientas, tutor: Tutor, registrador: Registrador):
    """Construye y compila el grafo integrado del asistente."""
    graph = StateGraph(AgentState)

    # Nodo principal de tutor
    graph.add_node("tutora", lambda state: nodo_tutora(state, llm_con_herramientas, tutor))
    # Nodo de herramientas
    graph.add_node("tools", ToolNode(lista_herramientas))
    # Nodo registrador
    graph.add_node("registrador", lambda state: nodo_registrador(state, registrador))

    # Flujo
    graph.set_entry_point("tutora")
    graph.add_conditional_edges(
        "tutora", should_continue, {"tools": "tools", "__end__": "registrador"}
    )
    graph.add_edge("tools", "tutora")
    graph.add_edge("registrador", END)

    print("🧩 Grafo del asistente 'Luna' construido e integrado con Tutor y Registrador.")
    return graph


# --- 5️⃣ Ejecución principal ---
def main():
    print("🧩 Iniciando setup...")  
    setup_environment()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )

    # Inicializamos agentes
    tutor = Tutor(llm)
    registrador = Registrador()
    print("✅ Entorno configurado correctamente.")

    # Herramientas
    herramientas = definir_herramientas()
    llm_con_herramientas = llm.bind_tools(herramientas)

    # Grafo integrado
    graph = construir_grafo(llm_con_herramientas, herramientas, tutor, registrador)

    # Checkpoints SQLite
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("🌍 Asistente de idiomas iniciado con LangGraph 🌍")
        historial = []

        while True:
            entrada = input("👤 Usuario: ")
            if entrada.lower() in ["salir", "exit", "quit"]:
                print("\n👋 Luna: ¡Hasta luego! Sigue practicando 🌟")
                break

            historial.append(HumanMessage(content=entrada))
            final_state = app.invoke({"messages": historial}, config={"configurable": {"thread_id": "chat_unico"}})
            historial = final_state["messages"]
            print(f"\n🤖 Luna: {historial[-1].content}\n")

    checkpointer.conn.close()


if __name__ == "__main__":
    main()
# python -m asistente_idiomas.main

#para push
#Ejecutas en la terminal: git add .
#git commit -m "Mensaje descriptivo sobre los nuevos cambios"
#git push origin main