# asistente_idiomas/agentes/registrador.py
from asistente_idiomas.tools.notion_tool import guardar_en_notion


class Registrador:
    """Agente responsable de registrar en Notion los aprendizajes o resultados."""

    def registrar(self, texto: str) -> str:
        """Guarda el texto en Notion y devuelve el resultado."""
        try:
            resultado = guardar_en_notion(texto)
            return f"✅ Registro guardado en Notion correctamente: {resultado}"
        except Exception as e:
            return f"⚠️ Error al guardar en Notion: {e}"
