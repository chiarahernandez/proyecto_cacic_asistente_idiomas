# asistente_idiomas/agentes/tutor.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from asistente_idiomas.tools.rag_tool import buscar_vocabulario
from dotenv import load_dotenv


class Tutor:
    """Agente tutor que conversa con el usuario y usa el RAG para responder."""

    def __init__(self, llm):
        # Recibimos el LLM desde afuera
        self.llm = llm

    def responder(self, pregunta: str) -> dict:
        """
        Procesa la pregunta del usuario, genera una respuesta y extrae
        el texto específico que se debe guardar en Notion.
        """
        # Palabras clave para detectar la intención de registrar
        keywords_registrar = ["registra", "guarda", "anota", "apunta"]
        
        texto_para_guardar = None
        
        # Lógica para detectar si se debe guardar algo
        if any(keyword in pregunta.lower() for keyword in keywords_registrar):
            # Creamos un prompt específico para que la IA extraiga la información
            prompt_extraccion = [
                SystemMessage(content="Tu tarea es extraer la palabra o frase que el usuario quiere guardar y su significado. Responde únicamente con el texto a guardar en formato 'palabra: significado'."),
                HumanMessage(content=pregunta)
            ]
            respuesta_extraccion = self.llm.invoke(prompt_extraccion).content
            texto_para_guardar = respuesta_extraccion.strip()

        # Generamos la respuesta conversacional para el usuario
        prompt_conversacion = [
            SystemMessage(content="Eres 'Luna', una tutora de idiomas amable y paciente."),
            HumanMessage(content=pregunta)
        ]
        respuesta_conversacional = self.llm.invoke(prompt_conversacion).content

        return {
            "respuesta": respuesta_conversacional,
            "texto_para_guardar": texto_para_guardar  # Puede ser None si no hay nada que guardar
        }