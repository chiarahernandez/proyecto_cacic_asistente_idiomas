# asistente_idiomas/agentes/tutor.py

import os
import json
import re
from datetime import datetime
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from asistente_idiomas.tools.rag_idioma import buscar_vocabulario


class Tutor:
    """
    Agente tutor que conversa con el usuario y usa RAG para responder.
    Mantiene un saludo inicial solo la primera vez y gestiona el historial.
    """

    def __init__(self, llm):
        self.llm = llm
        self.saludado = False
        self.historial_mensajes = []

        self.system_prompt = SystemMessage(
            content=(
                "Eres 'Luna', una profesora de idiomas formal, paciente y profesional. "
                "Detecta qué idioma el estudiante desea practicar. "
                "Responde en ese idioma y siempre incluye la traducción al español entre paréntesis."
            )
        )
        self.historial_mensajes.append(self.system_prompt)

    # 👇 FUNCIÓN PRINCIPAL
    def responder(self, pregunta: str) -> dict:
        """
        Procesa la pregunta del usuario, genera una respuesta y extrae el texto
        específico que se debe guardar en Notion.
        """
        keywords_registrar = ["registra", "guarda", "anota", "apunta"]
        texto_para_guardar = None
        respuesta_final = ""

        # --- 1️⃣ Primer saludo ---
        if not self.saludado:
            respuesta_final = (
                "¡Hola! Soy Luna, tu asistente de idiomas virtual. "
                "Estoy aquí para ayudarte a aprender vocabulario, frases y expresiones "
                "del idioma que quieras practicar. "
                "Recuerda que siempre te daré la traducción al español.\n\n"
                "Para empezar, ¿qué idioma te gustaría practicar hoy? "
                "(por ejemplo: inglés, francés o español)\n"
            )
            self.saludado = True
            return {"respuesta": respuesta_final, "texto_para_guardar": None}

        # --- 2️⃣ Guardamos el mensaje ---
        self.historial_mensajes.append(HumanMessage(content=pregunta))

        # --- 2.5️⃣ Detección flexible de consultas tipo "qué significa", etc. ---
        patrones_significado = [
            r"qué\s+significa",
            r"definición\s+de",
            r"significado\s+de",
            r"qué\s+quiere\s+decir",
            r"qué\s+es\s+",
            r"explica\s+la\s+palabra",
            r"explícame\s+la\s+palabra",
            r"dime\s+qué\s+significa",
        ]

        if any(re.search(p, pregunta.lower()) for p in patrones_significado):
            # Extraemos una posible palabra clave
            match_palabra = re.search(r"(significa|de|es)\s+([a-zA-Záéíóúüñ'-]+)", pregunta.lower())
            if match_palabra:
                palabra_consulta = match_palabra.group(2).strip(" ?")
            else:
                palabra_consulta = pregunta.split()[-1].strip(" ?")

            print(f"🔍 Buscando en RAG: {palabra_consulta}")

            # ✅ 1️⃣ Buscar en el RAG
            resultados_rag = buscar_vocabulario(palabra_consulta)

            # Si no encuentra nada, avisar
            if not resultados_rag:
                respuesta_final = (
                    f"No encontré información sobre '{palabra_consulta}' en mi base de conocimiento. "
                    "Podría ser una palabra inventada o un término fuera de mi base."
                )
                return {"respuesta": respuesta_final, "texto_para_guardar": None}

            # ✅ 2️⃣ Construir un contexto a partir del RAG
            contexto_rag = "\n\n".join(resultados_rag)
            prompt = (
                f"El usuario preguntó por la palabra '{palabra_consulta}'. "
                f"Usa la siguiente información del RAG para responder:\n\n"
                f"{contexto_rag}\n\n"
                f"Explica su significado de forma breve, tradúcela al español, y da un ejemplo de uso."
            )

            # ✅ 3️⃣ Llamar al modelo con ese contexto
            respuesta_llm = self.llm.invoke([HumanMessage(content=prompt)])

            respuesta_final = respuesta_llm.content
            texto_para_guardar = f"{palabra_consulta}: {respuesta_final}"

            return {"respuesta": respuesta_final, "texto_para_guardar": texto_para_guardar}

        # --- 3️⃣ Si el usuario pide registrar algo ---
        if any(keyword in pregunta.lower() for keyword in keywords_registrar):
            contexto = (
                f"La última palabra explicada fue '{self.ultima_palabra}'."
                if hasattr(self, "ultima_palabra")
                else "No hay palabra previa registrada."
            )

            # Obtenemos también el último mensaje del tutor para contexto
            ultimo_significado = ""
            if len(self.historial_mensajes) >= 2:
                ultimo_significado = self.historial_mensajes[-2].content

            print(f"DEBUG última palabra guardada: {getattr(self, 'ultima_palabra', None)}")

            # --- 4️⃣ Prompt reforzado para extracción JSON ---
            prompt_extraccion = [
                SystemMessage(
                    content=(
                        "Eres el 'Registrador', un asistente encargado de guardar en Notion "
                        "las palabras, frases o expresiones que el estudiante aprendió.\n\n"
                        "Tu tarea es analizar el contexto y devolver EXCLUSIVAMENTE un JSON VÁLIDO, "
                        "sin texto adicional.\n\n"
                        "El JSON debe contener las siguientes claves:\n"
                        "{'palabra', 'traduccion', 'ejemplo', 'idioma', 'fecha'}.\n\n"
                        "Ejemplo válido:\n"
                        "{'palabra': 'moon', 'traduccion': 'luna', "
                        "'ejemplo': 'The moon is bright tonight.', 'idioma': 'inglés', 'fecha': '2025-10-18'}\n\n"
                        "⚠️ Si NO puedes determinar con claridad qué palabra o frase se debe registrar, "
                        "NO inventes nada. En su lugar, responde exactamente con este mensaje:\n"
                        '"Necesito que me confirmes qué palabra o frase quieres registrar antes de guardar en Notion."'
                    )
                ),
                HumanMessage(
                    content=(
                        f"Usuario dijo: {pregunta}\n"
                        f"Contexto: {contexto}\n"
                        f"Última explicación del tutor: {ultimo_significado}\n"
                    )
                ),
            ]

            # --- 5️⃣ Llamamos al modelo para generar el JSON ---
            resultado = self.llm.invoke(prompt_extraccion)

            # --- 6️⃣ Intentamos parsear el JSON ---
            try:
                contenido = resultado.content if hasattr(resultado, "content") else str(resultado)
                print("\nDEBUG respuesta cruda del modelo:\n", contenido)

                contenido_limpio = re.sub(r"```json|```", "", contenido).strip()
                datos = json.loads(contenido_limpio.replace("'", '"'))
                print("DEBUG JSON decodificado correctamente:", datos)
            except Exception as e:
                print("⚠️ No se pudo convertir la respuesta a JSON.")
                print("🧾 Respuesta recibida del modelo:", resultado)
                print("⚠️ Error JSON:", e)
                datos = None

            texto_para_guardar = datos

            # --- 7️⃣ Continuación normal de la conversación ---
            prompt_conversacion = self.historial_mensajes
            resultado_conversacion = self.llm.invoke(prompt_conversacion)
            respuesta_texto = (
                resultado_conversacion.content.strip()
                if hasattr(resultado_conversacion, "content")
                else str(resultado_conversacion).strip()
            )

            self.historial_mensajes.append(AIMessage(content=respuesta_texto))
            respuesta_final = respuesta_texto

            # --- 8️⃣ Retornamos resultado y datos para registrar ---
            return {"respuesta": respuesta_final, "texto_para_guardar": texto_para_guardar}

        # --- 9️⃣ Si no es un caso especial, conversación normal ---
        resultado_generico = self.llm.invoke(self.historial_mensajes)
        respuesta_final = (
            resultado_generico.content.strip()
            if hasattr(resultado_generico, "content")
            else str(resultado_generico).strip()
        )
        self.historial_mensajes.append(AIMessage(content=respuesta_final))

        return {"respuesta": respuesta_final, "texto_para_guardar": None}
