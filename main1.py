from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import StateGraph

load_dotenv()

app = FastAPI(title="Agentic AI API")


@app.get("/health")
def health_check():
    return {"status": "ok"}