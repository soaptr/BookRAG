from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.chroma import get_vectorstore
from src.llm import get_qa_chain


# Создаем FastAPI приложение
app = FastAPI()

# Инициализация
vectorstore = get_vectorstore()
qa_chain = get_qa_chain(vectorstore, rerank='cross')

# Местоположение шаблонов
templates = Jinja2Templates(directory="templates")


# Модель для запроса
class QuestionRequest(BaseModel):
    question: str


# Эндпоинт для основной страницы с формой
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Эндпоинт для обработки запроса
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    result = qa_chain.invoke(question)
    answer = result.get("result", "Ответ не найден.")
    sources = result.get("source_documents", [])

    # Формирование списка источников
    source_list = []
    for source in sources:
        book_title = source.metadata.get('book_title', 'Unknown')
        chapter_title = source.metadata.get('chapter_title', 'Unknown')
        source_list.append(f"{book_title}, {chapter_title}")

    return {"answer": answer, "sources": source_list}
