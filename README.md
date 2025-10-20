# Asistente de Idiomas — Luna
##  Instalación y configuración

##  Clonar el repositorio

```bash
git clone https://github.com/chiarahernandez/proyecto_cacic_asistente_idiomas 
```

## Crear y activar entorno virtual:
python -m venv .venv

- Windows:

.venv\Scripts\Activate.ps1


- Mac / Linux:

source .venv/bin/activate

## Instalar dependencias:
pip install -r requirements.txt


## Configurar claves de API

Crear un archivo .env en la raíz del proyecto con las siguientes líneas:

- Clave API de Google Gemini
GOOGLE_API_KEY=tu_clave_gemini_aquí

- Token de Notion
NOTION_API_KEY=tu_clave_notion_aquí

- ID de la base de Notion donde se guardarán notas/vocabulario
NOTION_DATABASE_ID=tu_database_id_aquí

- API key de LangSmith (para trazabilidad y evaluación)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=tu_clave_langsmith_aquí
LANGCHAIN_PROJECT="Luna-Asistente-Idiomas"


## Ejecución

Ejecutar el asistente desde la raíz del proyecto:

python -m asistente_idiomas.main


**Luna** es un asistente virtual inteligente diseñado para ayudar a los usuarios a aprender, practicar y consultar vocabulario en distintos idiomas.  
- Combina técnicas avanzadas de Inteligencia Artificial conversacional, RAG (Retrieval-Augmented Generation) y el modelo Gemini (Google Generative AI), integradas mediante LangChain y LangGraph, para ofrecer una experiencia educativa natural, contextual y personalizada.


## Funcionalidades principales

- Tutor de idiomas: Luna actúa como tutora virtual que conversa, traduce y explica vocabulario en distintos idiomas.
- RAG (Búsqueda semántica): cuando el usuario consulta una palabra o frase corta, Luna busca en una base de documentos vectorizados para ofrecer su significado, traducción y ejemplos.
- Conversación libre: si la entrada del usuario es más extensa, Luna utiliza el modelo generativo Gemini para responder de forma contextual y natural.
- Detección de temas fuera de contexto: si el usuario pregunta algo no relacionado con idiomas, Luna responde amablemente que solo puede ayudar con temas lingüísticos.
- Vectorstore local persistente: utiliza **ChromaDB** para almacenar embeddings del vocabulario, acelerando las búsquedas futuras.
- Arquitectura modular: los componentes están organizados en carpetas (`tools/`, `graph_luna.py`, `main.py`) siguiendo buenas prácticas de software.
- Flujo orquestado con LangGraph: el agente está estructurado como un grafo con nodos definidos, representando el flujo de pensamiento del asistente.
- Guardar información o resúmenes en **Notion** como base de conocimiento.
- Registrar interacciones y flujos conversacionales con **LangSmith**, para análisis y mejora del modelo.


##  Arquitectura general

Usuario → Entrada de texto
↓
Nodo "Tutor" → decide si la consulta es corta o extensa
↓
Consulta corta →  Herramienta RAG  
(palabra/frase)  (búsqueda de vocabulario)

↓
Respuesta → Nodo "Luna"
↓
(Respuesta generada por Gemini o por RAG)

                   ┌─────────────────────────────────────┐
                   │          Usuario (CLI / Chat)       │
                   └─────────────────────────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────────┐
                      │    LangGraph: Graph Luna     │
                      ├─────────────────────────────┤
                      │ - Nodos: Lógica de diálogo   │
                      │ - Estado: Mensajes y contexto│
                      └─────────────────────────────┘
                                    │
               ┌──────────────────────────────┬──────────────────────────────┐
               ▼                              ▼                              ▼
       Google Gemini API              RAG Tool (Chroma)              Notion Integration
   (Generación de texto e IA)     (Embeddings y búsquedas)     (Notas, vocabulario, resúmenes)
                                    │
                                    ▼
                            LangSmith (Tracing)
                  (Registro y evaluación de conversaciones)


