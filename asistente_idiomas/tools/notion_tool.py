import os
from notion_client import Client
from datetime import datetime
from dotenv import load_dotenv

# 🧩 Carga las variables del entorno (.env)
load_dotenv()

# ✅ Inicializa el cliente de Notion
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)


def guardar_en_notion(palabra: str, traduccion: str, ejemplo: str, idioma: str, fecha: str):
    """
    Guarda un registro completo en Notion con las propiedades:
    Palabra, Traducción, Ejemplo de uso, Idioma, Fecha de estudio.
    """
    try:
        # Si la fecha viene como string, convertirla a formato ISO
        fecha_iso = datetime.fromisoformat(fecha).date().isoformat() if fecha else datetime.now().date().isoformat()

        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Palabra": {"title": [{"text": {"content": palabra[:2000]}}]},
                "Traducción": {"rich_text": [{"text": {"content": traduccion[:2000]}}]},
                "Ejemplo de uso": {"rich_text": [{"text": {"content": ejemplo[:2000]}}]},
                "Idioma": {"rich_text": [{"text": {"content": idioma[:2000]}}]},
                "Fecha": {"date": {"start": fecha_iso}},
            }
        )
        return f"✅ Registro guardado en Notion (ID: {response['id']})"
    except Exception as e:
        print(f"\nDEBUG: Error al guardar en Notion -> {e}\n")
        return f"❌ Error al guardar en Notion: {e}"
