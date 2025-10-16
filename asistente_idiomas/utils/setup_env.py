"""
utils/setup_environment.py

Este módulo se encarga de:
1. Cargar las variables de entorno desde un archivo .env.
2. Verificar que todas las claves necesarias estén definidas.
3. Mostrar mensajes informativos para confirmar que la configuración está correcta.

Se utiliza al inicio del main.py para preparar el entorno del asistente de idiomas.
"""

import os
from dotenv import load_dotenv


def setup_environment():
    """Carga las variables de entorno requeridas y valida su presencia."""

    # 1. Cargar las variables del archivo .env
    load_dotenv()

    # 2. Definir las variables que deben existir para el proyecto
    required_vars = [
        "GEMINI_API_KEY",       # para el modelo generativo y embeddings
        "LANGSMITH_API_KEY",    # para trazas y observabilidad
        "NOTION_API_KEY",       # para registrar resultados en Notion
        "NOTION_DATABASE_ID"    # ID de la base o página donde se guardarán los informes
    ]

    # 3. Verificar una por una si están presentes en el entorno
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("⚠️  Faltan las siguientes variables de entorno:")
        for var in missing_vars:
            print(f"   - {var}")
        raise ValueError("Configura correctamente el archivo .env antes de continuar.")

    print("✅ Entorno configurado correctamente. Todas las variables están definidas.")


if __name__ == "__main__":
    # Si ejecutás este archivo directamente, prueba la configuración
    setup_environment()
