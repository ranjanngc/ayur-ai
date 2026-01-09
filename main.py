# main.py - Charaka Samhita RAG API with WebSocket Streaming

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List
from operator import itemgetter
import json

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===========================
# CONFIG
# ===========================
JSONL_FILE = "charaka_clean-v1.jsonl"
DB_DIR = "chroma_charaka"
COLLECTION_NAME = "charaka"
K = int(os.getenv('TOKEN_LIMIT',5))
MODEL = "phi3:mini"  # Change to "phi3:mini" if gemma2 is too heavy

# ===========================
# Build DB once if not exists
# ===========================
def build_db_once():
    print("Building vector database (one-time only)...")
    texts = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for msg in data.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").strip()
                    if len(content) > 80:
                        texts.append(content)
    print(f"Loaded {len(texts)} passages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = splitter.create_documents(texts)
    print(f"Created {len(documents)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    ).persist()
    print("Vector DB built and saved!")

if not os.path.exists(DB_DIR):
    build_db_once()

# Load DB
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
retriever = vectordb.as_retriever(search_kwargs={"k": K})

# LLM & Prompt
# llm = ChatOllama(model=MODEL, temperature=0.1, num_predict=1024)
# llm = ChatOllama(
#     model=MODEL,
#     base_url="https://open-webui-production-4562.up.railway.app",  # Remote Ollama URL
#     temperature=0.1,
# )
# https://openrouter.ai/settings/keys

openAI_model=os.getenv('OPEN_AI_MODEL', 'meta-llama/llama-3.2-3b-instruct:free')
openAI_base_url=os.getenv('OPEN_AI_BASE_URL', 'https://openrouter.ai/api/v1')
openAI_api_key=os.getenv('OPEN_AI_KEY')
openAI_temperature=float(os.getenv('OPEN_AI_TEMPERATURE', 0.1))

print(f'openAI_model: {openAI_model}')
print(f'openAI_base_url: {openAI_base_url}')
print(f'openAI_api_key: {openAI_api_key}')
print(f'openAI_temperature: {openAI_temperature}')
print(f'K: {K}')
llm = ChatOpenAI(
    model=openAI_model,
    base_url="https://openrouter.ai/api/v1",
    api_key=openAI_api_key,  # Add as Railway secret
    temperature=openAI_temperature
)
prompt = PromptTemplate.from_template(
    """You are an expert Ayurvedic scholar deeply knowledgeable in Charaka Samhita.
Answer thoroughly using only the provided context.

CRITICAL RULES - MUST FOLLOW STRICTLY:
- Include a brief introduction referencing the provided context
- NEVER repeat sentences, paragraphs, or ideas — say each thing only once
- Do NOT summarize or rephrase the same information multiple times
- Be concise yet complete — no redundancy
- Use **bold** for headings, *italics* for Sanskrit terms, bullet points for lists
- Always put two blank lines between sections
- Include process to create medicine when required

Context:
{context}

Question: {question}

Detailed Answer:"""
)

# RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
    "sources": retriever
} | RunnableParallel({
    "answer": prompt | llm | StrOutputParser(),
    "sources": itemgetter("sources"),
    "question": itemgetter("question")
})

# FastAPI App
app = FastAPI(title="Charaka Samhita AI Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            question = data.get("question", "").strip()
            if not question:
                continue

            await websocket.send_json({"type": "status", "message": "Searching Charaka Samhita..."})

            result = rag_chain.invoke(question)
            full_answer = result["answer"]
            source_docs = result["sources"]

            sources = [
                {"index": i+1, "text": doc.page_content.strip()[:500] + ("..." if len(doc.page_content) > 500 else "")}
                for i, doc in enumerate(source_docs)
            ]

            await websocket.send_json({"type": "start", "question": question})

            words = full_answer.split()
            for word in words:
                await websocket.send_json({"type": "token", "token": word + " "})

            await websocket.send_json({
                "type": "end_rag",
                "answer": full_answer,
                "sources": sources
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})