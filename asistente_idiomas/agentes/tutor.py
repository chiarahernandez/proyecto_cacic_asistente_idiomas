# asistente_idiomas/agentes/tutor.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class Tutor:
    """
    Agente tutor que conversa con el usuario y usa RAG para responder.
    Además mantiene un saludo inicial solo la primera vez y GESTIONA EL HISTORIAL.
    """

    def __init__(self, llm):
        # Recibimos el LLM desde afuera
        self.llm = llm
        self.saludado = False  # Para controlar el saludo inicial
        # Inicializamos el historial de mensajes
        self.historial_mensajes = [] 

        # MODIFICACIÓN CLAVE: Instrucción estricta para la traducción
        self.system_prompt = SystemMessage(
            content=(
                "Eres 'Luna', una profesora de idiomas formal, paciente y profesional. "
                "Detecta qué idioma el estudiante desea practicar (por ejemplo inglés, francés o español). "
                "Tu tarea es responder casi completamente en ese idioma."
                "SIN EXCEPCIÓN, después de cada frase o unidad de texto que escribas en el idioma de práctica,"
                "DEBES incluir su traducción completa al español, encerrada entre paréntesis."
                "No uses español fuera de las traducciones entre paréntesis, a menos que sea una aclaración muy necesaria."
                "El formato estricto debe ser: 'Frase en idioma extranjero (Traducción en español)'."
                "Adapta el nivel al estudiante y da ejemplos prácticos."
            )
        )
        # Añadimos el SystemMessage al historial para que el LLM lo sepa siempre
        self.historial_mensajes.append(self.system_prompt)


    def responder(self, pregunta: str) -> dict:
        """
        Procesa la pregunta del usuario, genera una respuesta y extrae
        el texto específico que se debe guardar en Notion.
        """

        # Palabras clave para detectar la intención de registrar
        keywords_registrar = ["registra", "guarda", "anota", "apunta"]

        texto_para_guardar = None
        respuesta_final = ""

        #  Saludo inicial solo la primera vez
        if not self.saludado:
            # NOTA: El saludo inicial se mantiene solo en español.
            respuesta_final = ( 
                "¡Hola! Soy Luna, tu asistente de idiomas virtual. "
                "Estoy aquí para ayudarte a aprender vocabulario, frases y expresiones del idioma que quieras practicar. "
                "Recuerda que siempre te daré la traducción al español.\n\n"
                "Para empezar, ¿qué idioma te gustaría practicar hoy? (por ejemplo: inglés, francés o español)\n"
            )
            self.saludado = True
            
            # Devolver solo el saludo en la primera interacción y salir
            return {
                "respuesta": respuesta_final,
                "texto_para_guardar": None 
            }
        
        #  Añadir el mensaje del usuario al historial
        self.historial_mensajes.append(HumanMessage(content=pregunta))

        # ---------------------------------------------------------------
        # El resto del código solo se ejecuta DESPUÉS del primer saludo
        
        # --- Detectar si es una pregunta para guardar ---
        if any(keyword in pregunta.lower() for keyword in keywords_registrar):
            # El prompt de extracción es una tarea separada y NO debe llevar la instrucción de traducción forzada.
            prompt_extraccion = [
                SystemMessage(
                    content=(
                        "Tu tarea es extraer la palabra o frase que el usuario quiere guardar y su significado. "
                        "Responde únicamente con el texto a guardar en formato 'palabra: significado'."
                    )
                ),
                HumanMessage(content=pregunta)
            ]

            resultado = self.llm.invoke(prompt_extraccion)

            # Protección si el resultado no tiene atributo .content
            if hasattr(resultado, "content"):
                texto_para_guardar = resultado.content.strip()
            else:
                texto_para_guardar = str(resultado).strip()

        # --- Generar respuesta conversacional ---
        
        # El prompt de conversación ahora es el historial completo, que incluye el SystemMessage modificado.
        prompt_conversacion = self.historial_mensajes 
        
        resultado_conversacion = self.llm.invoke(prompt_conversacion)

        if hasattr(resultado_conversacion, "content"):
            respuesta_texto = resultado_conversacion.content.strip()
        else:
            respuesta_texto = str(resultado_conversacion).strip()
        
        # 🆕 Añadir la respuesta de la IA al historial
        self.historial_mensajes.append(AIMessage(content=respuesta_texto))

        # La respuesta final es la respuesta del LLM
        respuesta_final = respuesta_texto

        return {
            "respuesta": respuesta_final,
            "texto_para_guardar": texto_para_guardar  # Puede ser None si no hay nada que guardar
        }
        