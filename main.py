from src.chroma import get_vectorstore
from src.llm import get_qa_chain
import warnings


warnings.filterwarnings("ignore")


if __name__ == '__main__':
    vectorstore = get_vectorstore()
    qa_chain = get_qa_chain(vectorstore)

    while True:
        question = input("Ваш вопрос: ")
        if question.lower() in ['exit', 'выход']:
            break

        result = qa_chain.invoke(question)
        answer = result.get("result", "Ответ не найден.")
        sources = result.get("source_documents", [])

        print(f"\nОтвет: {answer}\n")

        if sources:
            print("Использованные источники:")
            for i, source in enumerate(sources, start=1):
                print(f"{i}. {source.metadata.get('book_title')}, {source.metadata.get('chapter_title')}")
