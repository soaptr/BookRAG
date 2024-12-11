import requests
from bs4 import BeautifulSoup
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


BASE_URL = "http://flibusta.site/b/"
BOOK_IDS = [170876, 167232, 167334, 167233, 167333]


def download_and_parse_book(book_id):
    """
    Скачивает и парсит книгу с указанным ID.

    Args:
        book_id (int): ID книги.

    Returns:
        list: Список документов, представляющих главы книги.
    """
    url = f"{BASE_URL}{book_id}/read"
    print(f"Скачиваем книгу с URL: {url}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Ошибка при загрузке книги {book_id}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    book_title = soup.find("h1", class_="title").get_text(strip=True) if soup.find("h1", class_="title") else f"{book_id}"
    book_title = re.sub(r'\s*\(.*?\)', '', book_title).strip()
    chapter_documents = []
    chapters = soup.find_all("h3", class_="book")

    for i, chapter in enumerate(chapters):
        chapter_title = chapter.get_text(strip=True)

        content = []
        for sibling in chapter.find_next_siblings():
            if sibling.name == "h3" and "book" in sibling.get("class", []):
                break
            if sibling.name == "p" and "book" in sibling.get("class", []):
                content.append(sibling.get_text(strip=True))

        chapter_content = "\n".join(content)

        chapter_documents.append(
            Document(
                page_content=chapter_content,
                metadata={
                    "chapter_title": chapter_title,
                    "book_title": book_title,
                }
            )
        )

    return chapter_documents


def get_chapters():
    """
    Извлекает главы из всех книг, указанных в глобальном списке BOOK_IDS.

    Returns:
        list: Список всех глав из указанных книг. Каждая глава представлена в формате,
        возвращаемом функцией `download_and_parse_book`.
    """
    chapters = []
    for book_id in BOOK_IDS:
        new_chapters = download_and_parse_book(book_id)
        chapters.extend(new_chapters)
    return chapters


def get_splits(chunk_size=1000, chunk_overlap=100):
    """
    Разбивает текст глав книг на части заданного размера.

    Args:
        chunk_size (int, optional): Размер одного чанка в символах. По умолчанию 1000.
        chunk_overlap (int, optional): Количество символов, перекрывающихся между чанками. По умолчанию 100.

    Returns:
        list: Список разделенных частей текста (чанков), где каждая часть представляет собой фрагмент
        оригинального текста главы.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(get_chapters())
    return splits
