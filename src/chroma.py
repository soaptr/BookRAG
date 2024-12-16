import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.flibusta import get_splits


def get_emb_model(model_name="intfloat/multilingual-e5-large-instruct"):
    """
    Создает и возвращает модель эмбеддингов.

    Args:
        model_name (str, optional): Название модели эмбеддингов

    Returns:
        HuggingFaceEmbeddings: Объект модели для генерации эмбеддингов.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstore(persist_directory="./chroma_db", need_update=False):
    """
    Создает или загружает хранилище векторных данных (vectorstore).

    Args:
        persist_directory (str, optional): Путь для сохранения или загрузки векторного хранилища.
        need_update (bool, optional): Указывает, нужно ли обновить хранилище.

    Returns:
        Chroma: Объект хранилища векторных данных.
    """
    if not os.path.exists(persist_directory) or need_update:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        vectorstore = Chroma.from_documents(
            documents=get_splits(),
            embedding=get_emb_model(),
            persist_directory=persist_directory
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            embedding_function=get_emb_model(),
            persist_directory=persist_directory
        )
    return vectorstore
