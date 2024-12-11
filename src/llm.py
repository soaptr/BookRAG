import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def get_qa_chain(vectorstore, top_k=30):
    """
    Создает QA Chain для обработки запросов по сюжету и персонажам книг.

    Args:
        vectorstore (Chroma): Векторное хранилище, используемое для поиска релевантных фрагментов текста.
        top_k (int, optional): Количество релевантных фрагментов, которые будут извлечены из хранилища
                               для ответа на запрос. По умолчанию 30.

    Returns:
        RetrievalQA: Объект QA Chain, который может обрабатывать запросы.
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    template = """
        Вы - помощник для решения вопросов пользователя по сюжету и персонажам книг.
        Используйте следующие фрагменты найденного контекста, чтобы ответить на вопрос.
        Если вы не знаете ответа, просто скажите, что не знаете.
        При ответе на вопрос используйте цитаты из книги.
        Вопрос: {question}
        Контекст: {context}
        Ответ:
        """
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain
