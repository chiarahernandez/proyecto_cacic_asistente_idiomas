from asistente_idiomas.tools.notion_tool import guardar_en_notion
from datetime import datetime

class Registrador:
    """Agente responsable de registrar en Notion los aprendizajes o resultados."""

    def registrar(self, datos):
        """Guarda los datos en Notion y devuelve el resultado."""
        try:
            if not isinstance(datos, dict):
                return "⚠️ Formato inválido: se esperaba un diccionario con los campos de Notion."

            palabra = datos.get("palabra", "")
            traduccion = datos.get("traduccion", "")
            ejemplo = datos.get("ejemplo", "")
            idioma = datos.get("idioma", "")
            
            #  Sobrescribimos siempre con la fecha actual
            fecha = datetime.now().date().isoformat()

            resultado = guardar_en_notion(palabra, traduccion, ejemplo, idioma, fecha)
            return f"✅ Registro guardado en Notion correctamente: {resultado}"
        except Exception as e:
            return f"⚠️ Error al guardar en Notion: {e}"
