from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

import google.generativeai as genai

import os

from rag import retrieve

load_dotenv()

genai.configure(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

app = FastAPI()

class ChatRequest(BaseModel):
    pergunta: str


@app.post("/chat")
def chat(req: ChatRequest):

    contexto = retrieve(
        req.pergunta
    )

    prompt = f"""
Você é um assistente da faculdade.

Responda SOMENTE usando
as informações abaixo.

Se não souber, informe.

Contexto:

{contexto}

Pergunta:

{req.pergunta}
"""

    response = model.generate_content(
        prompt
    )

    return {
        "resposta": response.text
    }