from src.chroma import get_vectorstore
from src.llm import get_qa_chain
from src.royallib import get_book_and_vector_store
import warnings


warnings.filterwarnings("ignore")


if __name__ == '__main__':

    author_name = 'Куприн Александр'
    book_name = 'Яма'

    vectorstore = get_book_and_vector_store(
        book_name=book_name,
        author_name=author_name,
        download_directory="./books_library/"
    )
    qa_chain = get_qa_chain(vectorstore, rerank='cross')

    # vectorstore = get_vectorstore()
    # qa_chain = get_qa_chain(vectorstore, rerank='cross')

    while True:
        question = input("Ваш вопрос: ")
        if question.lower() in ['exit', 'выход']:
            break

        result = qa_chain.invoke(question)
        answer = result.get("result", "Ответ не найден.")
        sources = result.get("source_documents", [])

        print(f"\nОтвет: {answer}\n")

        seen = set()
        source_count = 0
        print("Использованные источники:")
        for source in sources:
            # book_title = source.metadata.get('book_title', 'Unknown')
            # chapter_title = source.metadata.get('chapter_title', 'Unknown')
            headers = source.metadata.get("headers", [])
            print(headers)
            print("Контекст:", source.page_content)

            # unique_key = (book_title, chapter_title)

            # if unique_key not in seen:
            #     source_count += 1
            #     seen.add(unique_key)
            #     print(f"{source_count}. {book_title}, {chapter_title}")

        print("\n")
