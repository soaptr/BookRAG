import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


load_dotenv()
os.environ["GOOGLE_API_KEY"] = "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" #os.getenv("GOOGLE_API_KEY")


def get_qa_chain(vectorstore, top_k=30, rerank=None, rank_fraction=0.5):
    """
    Создает QA Chain для обработки запросов по сюжету и персонажам книг.

    Args:
        vectorstore (Chroma): Векторное хранилище, используемое для поиска релевантных фрагментов текста.
        top_k (int, optional): Количество релевантных фрагментов, которые будут извлечены из хранилища
                               для ответа на запрос. По умолчанию 30.
        rerank (str, optional): Метод повторного ранжирования. Поддерживаемые значения:
                                - 'cross': Использует Cross-Encoder для ранжирования.
                                - 'flash': Использует Flashrank для ранжирования.
                                - None: Без повторного ранжирования. По умолчанию None.
        rank_fraction (float, optional): Доля фрагментов, используемых после повторного ранжирования.
                                          Значение должно быть в диапазоне (0, 1]. По умолчанию 0.5.

    Returns:
        RetrievalQA: Объект QA Chain, который может обрабатывать запросы.
    """
    if not 0 < rank_fraction <= 1:
        raise ValueError("Параметр 'rank_fraction' должен быть в диапазоне (0, 1].")
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

    if rerank == 'cross':
        encoder = HuggingFaceCrossEncoder(model_name="DiTy/cross-encoder-russian-msmarco")
        compressor = CrossEncoderReranker(
            model=encoder,
            top_n=max(1, int(top_k * rank_fraction))
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"k": top_k})
        )
    elif rerank == 'flash':
        compressor = FlashrankRerank(
            top_n=max(1, int(top_k * rank_fraction))
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"k": top_k})
        )
    elif rerank is None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    else:
        raise ValueError("Недопустимое значение для параметра 'rerank'. Используйте 'cross', 'flash' или None.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain
