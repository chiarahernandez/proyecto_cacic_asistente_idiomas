# asistente_idiomas/main.py
import os
from typing import Sequence, Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# Tus agentes, que ahora usaremos correctamente
from asistente_idiomas.agentes.tutor import Tutor
from asistente_idiomas.agentes.registrador import Registrador

# Herramientas
from asistente_idiomas.tools.rag_idioma import buscar_vocabulario
from asistente_idiomas.tools.notion_tool import guardar_en_notion
from asistente_idiomas.tools.rag_idioma import off_topic_tool

# --- 1️⃣ Configuración de entorno ---
def setup_environment():
    """Carga variables desde el archivo .env y valida la API key."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("⚠️ Falta la variable GEMINI_API_KEY en tu archivo .env")
    print("✅ Variables de entorno cargadas correctamente.")


# --- 2️⃣ Definición de herramientas (sin cambios) ---
# (Tu código de herramientas @tool va aquí, lo omito por brevedad pero debe estar)
@tool
def herramienta_buscar_vocabulario(palabra: str):
    """Busca definiciones o traducciones de palabras en la base RAG de idiomas."""
    return buscar_vocabulario(palabra)

@tool
def herramienta_guardar_en_notion(texto_a_guardar: str):
    """
    Guarda una ÚNICA cadena de texto en la base de datos de Notion.
    Esta herramienta solo acepta un argumento de tipo string llamado 'texto_a_guardar'.
    Úsala para registrar la palabra y su significado juntos en una sola frase.
    Por ejemplo: 'yellow: amarillo'.
    NO intentes pasar argumentos separados para la palabra y el significado.
    """
    return guardar_en_notion(texto_a_guardar)

@tool
def herramienta_off_topic():
    """Responde cuando el usuario pregunta algo fuera del contexto educativo."""
    return off_topic_tool()

def definir_herramientas():
    print("🛠️ Herramientas cargadas: vocabulario, guardar_en_notion, off_topic.")
    return [
        herramienta_buscar_vocabulario,
        herramienta_guardar_en_notion,
        herramienta_off_topic,
    ]


# --- 3️⃣ Estado compartido (MODIFICADO) ---
class AgentState(TypedDict):
    """Estado compartido entre nodos del grafo."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Canal de comunicación para que el Tutor le diga al Registrador QUÉ guardar.
    texto_para_registrar: str | None


# --- 4️⃣ Nodos del grafo (CORREGIDOS) ---
def nodo_tutor(state: AgentState, tutor: Tutor):
    """
    El agente Tutor procesa el último mensaje del usuario, genera una respuesta
    y decide si algo debe ser registrado, pasándolo a través del estado.
    """
    last_user_message = state["messages"][-1].content
    
    # El tutor procesa la entrada y devuelve la respuesta y el texto a guardar
    resultado_tutor = tutor.responder(last_user_message)
    
    respuesta_para_usuario = resultado_tutor.get("respuesta", "No pude procesar tu solicitud.")
    texto_a_guardar = resultado_tutor.get("texto_para_guardar")

    # Devolvemos la respuesta del Tutor como un mensaje de la IA y actualizamos el estado
    return {
        "messages": [AIMessage(content=respuesta_para_usuario)],
        "texto_para_registrar": texto_a_guardar
    }

def nodo_registrador(state: AgentState, registrador: Registrador):
    """
    El agente Registrador toma el texto del estado y lo guarda en Notion.
    """
    texto = state.get("texto_para_registrar")
    if texto:
        print(f"✔️  Registrador: Recibí la orden de guardar '{texto}'.")
        resultado = registrador.registrar(texto)
        print(f"✔️  Registrador: {resultado}")
    
    # Limpiamos el estado para la siguiente ronda
    return {"texto_para_registrar": None}

def should_register(state: AgentState) -> Literal["registrador", "__end__"]:
    """Decide si el flujo debe ir al agente Registrador o terminar."""
    if state.get("texto_para_registrar"):
        return "registrador"
    else:
        return "__end__"


def construir_grafo(tutor: Tutor, registrador: Registrador):
    """Construye y compila el grafo con los agentes Tutor y Registrador."""
    graph = StateGraph(AgentState)

    graph.add_node("tutor", lambda state: nodo_tutor(state, tutor))
    graph.add_node("registrador", lambda state: nodo_registrador(state, registrador))

    graph.set_entry_point("tutor")
    graph.add_conditional_edges(
        "tutor",
        should_register,
        {
            "registrador": "registrador",
            "__end__": END,
        },
    )
    graph.add_edge("registrador", END)

    print("🧩 Grafo construido con agentes 'Tutor' y 'Registrador' en colaboración.")
    return graph


# --- 5️⃣ Ejecución principal (CORREGIDA) ---
def main():
    print("🧩 Iniciando setup...")  
    setup_environment()

    # Dejamos tu modelo original, como pediste
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )

    tutor = Tutor(llm)
    registrador = Registrador()
    print("✅ Entorno configurado correctamente.")

    graph = construir_grafo(tutor, registrador)

    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("🌍 Asistente de idiomas iniciado con LangGraph 🌍")
        historial = []

        while True:
            entrada = input("👤 Usuario: ")
            if entrada.lower() in ["salir", "exit", "quit"]:
                print("\n👋 Luna: ¡Hasta luego! Sigue practicando 🌟")
                break

            # Usamos el historial local que ya tenías
            historial.append(HumanMessage(content=entrada))
            
            # El checkpointer maneja la memoria entre ejecuciones
            config = {"configurable": {"thread_id": "chat_unico"}}
            
            # Invocamos el grafo
            final_state = app.invoke({"messages": historial}, config=config)
            
            # Agregamos la respuesta del AI al historial
            historial.append(final_state["messages"][-1])
            
            print(f"\n🤖 Luna: {final_state['messages'][-1].content}\n")

if __name__ == "__main__":
    main()

# python -m asistente_idiomas.main

#para push
# git remote add origin https://github.com/chiarahernandez/proyecto_cacic_asistente_idiomas

#Ejecutas en la terminal: git add .
#git commit -m "Mensaje descriptivo sobre los nuevos cambios"
#git push origin main