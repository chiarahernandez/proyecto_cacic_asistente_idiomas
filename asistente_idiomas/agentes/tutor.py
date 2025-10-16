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
        Procesa la pregunta del usuario.
        Si es sobre vocabulario, consulta el RAG y devuelve la respuesta.
        Si es relevante para registrar, marca la intención.
        """
        # Buscar en la base de conocimiento (RAG)
        resultado_rag = buscar_vocabulario(pregunta)

        prompt = [
            SystemMessage(content="Sos un tutor de idiomas paciente y claro."),
            HumanMessage(content=f"La consulta del usuario es: {pregunta}"),
            AIMessage(content=f"Información recuperada: {resultado_rag}"),
        ]

        respuesta = self.llm.invoke(prompt)

        # Decidir si hay que registrar el aprendizaje
        registrar = any(palabra in pregunta.lower() for palabra in ["nuevo", "aprender", "recordar"])

        return {
            "respuesta": respuesta.content,
            "registrar": registrar
        }
