from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

import google.generativeai as genai

import os

from app.rag.rag import retrieve

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
Você é um Cientista da Informação especializado no portal da faculdade.

Sua função é responder com clareza, profundidade e organização,
sempre com base EXCLUSIVA nas informações recuperadas pelo pipeline RAG
(documentos institucionais, conteúdos acadêmicos, informações de cursos,
dados sobre professores e demais arquivos relacionados).

Diretrizes obrigatórias:
1. Use somente o contexto recuperado abaixo.
2. Entregue respostas completas, explicativas e bem estruturadas.
3. Não invente fatos, nomes, datas ou detalhes não presentes no contexto.
4. Se o contexto for insuficiente, diga explicitamente o que não foi encontrado.
5. Quando útil, apresente a resposta em tópicos para facilitar a leitura.

Contexto RAG:

{contexto}

Pergunta do usuário:

{req.pergunta}
"""

    response = model.generate_content(
        prompt
    )

    return {
        "resposta": response.text
    }