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
        self.ultima_palabra = None  # 🧩 para evitar errores entre modos

        self.system_prompt = SystemMessage(
            content=(
                "Eres 'Luna', una profesora de idiomas formal, paciente y profesional. "
                "Detecta qué idioma el estudiante desea practicar. "
                "Responde en ese idioma y siempre incluye la traducción al español entre paréntesis."
            )
        )
        self.historial_mensajes.append(self.system_prompt)

    # 🧠 Función auxiliar para extraer la palabra consultada
    def _extraer_palabra(self, texto: str) -> str:
        """
        Extrae la palabra clave de una pregunta del tipo:
        'qué significa la palabra zorplin' -> 'zorplin'
        """
        texto = texto.lower().strip()
        texto = re.sub(
            r"(qué\s+significa|qué\s+quiere\s+decir|definición\s+de|significado\s+de|qué\s+es|explica\s+la\s+palabra|explícame\s+la\s+palabra|dime\s+qué\s+significa)",
            "",
            texto,
        )
        texto = re.sub(r"\b(la|el|una|un|palabra|de|del|es)\b", "", texto)
        match = re.findall(r"[a-záéíóúüñ'-]+", texto)
        return match[-1] if match else texto

    # 👇 FUNCIÓN PRINCIPAL
    def responder(self, pregunta: str) -> dict:
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

        # --- 2.5️⃣ Detección de consultas tipo "qué significa..." ---
        patrones_significado = [
            r"qu[eé]\s+significa",
            r"definici[oó]n\s+de",
            r"significado\s+de",
            r"qu[eé]\s+quiere\s+decir",
            r"qu[eé]\s+es\s+",
            r"explica\s+la\s+palabra",
            r"expl[ií]came\s+la\s+palabra",
            r"dime\s+qu[eé]\s+significa",
        ]

        if any(re.search(p, pregunta.lower()) for p in patrones_significado):
            palabra_consulta = self._extraer_palabra(pregunta)
            self.ultima_palabra = palabra_consulta

            print(f"🔍 Buscando en RAG: {palabra_consulta}")
            resultados_rag = buscar_vocabulario(palabra_consulta)

            if not resultados_rag:
                print(f"⚠️ '{palabra_consulta}' no encontrada en el RAG. Consultando directamente al modelo Gemini...")
                prompt_fallback = (
                    f"Define y traduce al español la palabra o expresión '{palabra_consulta}'. "
                    f"Explica brevemente su significado y da un ejemplo de uso en inglés."
                )
                respuesta_llm = self.llm.invoke([HumanMessage(content=prompt_fallback)])
                respuesta_final = respuesta_llm.content

                texto_para_guardar = f"{palabra_consulta}: {respuesta_final}"
                self.ultima_palabra = palabra_consulta
                return {"respuesta": respuesta_final, "texto_para_guardar": texto_para_guardar}

            contexto_rag = "\n\n".join(resultados_rag)
            prompt = (
                f"El usuario preguntó por la palabra '{palabra_consulta}'. "
                f"Usa la siguiente información del RAG para responder:\n\n"
                f"{contexto_rag}\n\n"
                f"Explica su significado de forma breve, tradúcela al español, y da un ejemplo de uso."
            )

            respuesta_llm = self.llm.invoke([HumanMessage(content=prompt)])
            respuesta_final = respuesta_llm.content
            texto_para_guardar = f"{palabra_consulta}: {respuesta_final}"

            # 🧹 Limpieza de contexto del RAG tras responder
            self.ultima_palabra = palabra_consulta  # se conserva para registro inmediato
            return {"respuesta": respuesta_final, "texto_para_guardar": texto_para_guardar}

        # --- 3️⃣ Registro en Notion ---
        if any(keyword in pregunta.lower() for keyword in keywords_registrar):
            contexto = (
                f"La última palabra explicada fue '{self.ultima_palabra}'." 
                if self.ultima_palabra else "No hay palabra previa registrada."
            )

            ultimo_significado = ""
            if len(self.historial_mensajes) >= 2:
                ultimo_significado = self.historial_mensajes[-2].content

            prompt_extraccion = [
                SystemMessage(
                    content=(
                        "Eres el 'Registrador', un asistente encargado de guardar en Notion "
                        "las palabras, frases o expresiones que el estudiante aprendió.\n\n"
                        "Tu tarea es analizar el contexto y devolver EXCLUSIVAMENTE un JSON VÁLIDO, "
                        "sin texto adicional.\n\n"
                        "El JSON debe contener las siguientes claves:\n"
                        "{'palabra', 'traduccion', 'ejemplo', 'idioma', 'fecha'}.\n\n"
                        "⚠️ Si NO puedes determinar con claridad qué palabra o frase se debe registrar, "
                        "NO inventes nada. En su lugar, responde exactamente con este mensaje:\n"
                        '"Necesito que me confirmes qué palabra o frase quieres registrar antes de guardar en Notion."'
                    )
                ),
                HumanMessage(
                    content=(f"Usuario dijo: {pregunta}\nContexto: {contexto}\nÚltima explicación del tutor: {ultimo_significado}\n")
                ),
            ]

            resultado = self.llm.invoke(prompt_extraccion)

            try:
                contenido = resultado.content if hasattr(resultado, "content") else str(resultado)
                print("\nDEBUG respuesta cruda del modelo:\n", contenido)
                contenido_limpio = re.sub(r"```json|```", "", contenido, flags=re.DOTALL).strip()
                datos = json.loads(contenido_limpio)
                print("✅ DEBUG JSON decodificado correctamente:", datos)
            except Exception as e:
                print("⚠️ No se pudo convertir la respuesta a JSON.")
                print("🧾 Respuesta recibida del modelo:", resultado)
                print("⚠️ Error JSON:", e)
                datos = None

            texto_para_guardar = datos
           # 💾 Confirmación directa de registro sin invocar nuevamente al modelo
            if datos:
                respuesta_final = f"✅ La palabra **{datos.get('palabra', '')}** fue registrada correctamente en Notion."
            else:
                respuesta_final = "⚠️ No se pudo registrar la palabra. Verifica que se haya reconocido correctamente."

            # 🧹 Limpieza tras registrar en Notion
            self.ultima_palabra = None
            return {"respuesta": respuesta_final, "texto_para_guardar": texto_para_guardar}

        # --- 9️⃣ Conversación normal ---
        resultado_generico = self.llm.invoke(self.historial_mensajes)
        respuesta_final = (
            resultado_generico.content.strip()
            if hasattr(resultado_generico, "content")
            else str(resultado_generico).strip()
        )
        self.historial_mensajes.append(AIMessage(content=respuesta_final))

        return {"respuesta": respuesta_final, "texto_para_guardar": None}
