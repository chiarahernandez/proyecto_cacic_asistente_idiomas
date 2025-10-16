# asistente_idiomas/tools/rag_tool.py
from typing import List, Dict

def buscar_vocabulario(palabra: str) -> Dict[str, str]:
    """
    Simula una búsqueda en la base de conocimientos de vocabulario.
    En la versión real, podría hacer una consulta RAG a un modelo o a Notion.
    """
    base_ejemplo = {
        "hello": {"traduccion": "hola", "ejemplo": "Hello, how are you?"},
        "world": {"traduccion": "mundo", "ejemplo": "The world is beautiful."},
        "language": {"traduccion": "idioma", "ejemplo": "Spanish is a beautiful language."}
    }

    palabra_lower = palabra.lower()
    if palabra_lower in base_ejemplo:
        return {
            "respuesta": f"{palabra} significa '{base_ejemplo[palabra_lower]['traduccion']}'. Ejemplo: {base_ejemplo[palabra_lower]['ejemplo']}",
            "registrar": True
        }

    return {"respuesta": f"No encontré la palabra '{palabra}'.", "registrar": False}
