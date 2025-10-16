# utils/test_notion.py
import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.getenv("NOTION_API_KEY"))
database_id = os.getenv("NOTION_DATABASE_ID")

# Intentar leer la base
try:
    response = notion.databases.retrieve(database_id=database_id)
    print(f"✅ Conexión exitosa a la base de Notion: {response['title'][0]['plain_text']}")
except Exception as e:
    print("❌ Error al conectar con Notion:")
    print(e)
