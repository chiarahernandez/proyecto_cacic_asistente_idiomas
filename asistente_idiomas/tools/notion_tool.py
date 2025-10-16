import os
from notion_client import Client
from datetime import datetime
from dotenv import load_dotenv

# 🧩 Carga las variables del entorno (.env)
load_dotenv()



# ✅ Inicializa el cliente de Notion
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
print("DEBUG - NOTION_TOKEN:", NOTION_TOKEN)
print("DEBUG - NOTION_DATABASE_ID:", NOTION_DATABASE_ID)


notion = Client(auth=NOTION_TOKEN)


# en asistente_idiomas/tools/notion_tool.py

def guardar_en_notion(texto: str):
    try:
        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Name": {
                    "title": [{"text": {"content": texto[:2000]}}]
                }
            }
        )
        return f"✅ Registro guardado en Notion (ID: {response['id']})"
    except Exception as e:
        # ¡Línea clave para debugging!
        print(f"\nDEBUG: El error completo de la API de Notion es: {e}\n") 
        return f"❌ Error al guardar en Notion: {e}"

