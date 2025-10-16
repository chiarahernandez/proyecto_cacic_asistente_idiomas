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


def guardar_en_notion(contenido: str):
    """
    Guarda un resumen, reporte o información clave en la base de datos de Notion.

    Parámetros:
    - contenido (str): texto que se desea guardar como entrada.

    Retorna:
    - str: mensaje de confirmación o error.
    """
    try:
        # Crea la página en la base de datos de Notion
        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Título": {
                    "title": [
                        {"text": {"content": f"Registro - {datetime.now().strftime('%d/%m/%Y %H:%M')}"}}
                    ]
                },
                "Contenido": {
                    "rich_text": [
                        {"text": {"content": contenido[:2000]}}  # límite seguro de caracteres
                    ]
                },
            },
        )
        return f"✅ Informe guardado correctamente en Notion (ID: {response['id']})"
    except Exception as e:
        return f"❌ Error al guardar en Notion: {str(e)}"
