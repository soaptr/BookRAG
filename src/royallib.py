import requests
from bs4 import BeautifulSoup
import os
import zipfile
import shutil
from tqdm import tqdm
import logging
from langchain.schema import Document
from langchain.vectorstores import Chroma
from src.chroma import get_emb_model



def download_book(book_name, author_name, download_directory="./books_library/"):

    url = 'https://royallib.com/search/'

    payload = {
        'to': 'result',
        'q': book_name
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()

    # Обработка ответа в кодировке utf-8
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('tr')

    matching_books = []
    for result in results:
        # Находим ссылку на книгу
        book_link = result.find('td').find('a')

        try:
            author_link = result.find_all('td')[2].find('a')

            if book_link and author_link:
                # Извлекаем название книги
                current_book_name = book_link.text.strip()

                # Извлекаем имя автора
                current_author_name = author_link.text.strip()

                # Проверяем соответствие названия и автора
                if (book_name.lower() in current_book_name.lower() and
                    author_name.lower() in current_author_name.lower()):

                    # Получаем полную ссылку на книгу
                    book_url = 'https:' + book_link['href']

                    matching_books.append({
                        'book_name': current_book_name,
                        'author_name': current_author_name,
                        'book_url': book_url
                    })
        except:
            pass

    if not matching_books:
        print("Книга не найдена")
        return None

    book_link = matching_books[0]['book_url']

    book_page = requests.get(book_link)
    book_page.encoding = 'utf-8'
    book_soup = BeautifulSoup(book_page.text, 'html.parser')

    results = book_soup.find_all('a')

    download_link = None
    for result in results:
        link_description = result.text.strip()
        if 'скачать в формате html' in link_description.lower():
            download_link = result['href']
            break

    if not download_link:
        print("Ссылка для скачивания HTML не найдена")
        return None

    # Полный URL-адрес
    full_url = f"https:{download_link}"

    # Создание директории для загрузки, если она не существует
    os.makedirs(download_directory, exist_ok=True)

    # Генерация пути для временного ZIP-файла
    zip_filename = os.path.basename(download_link)
    zip_path = os.path.join(download_directory, zip_filename)

    try:
        # Скачивание ZIP-файла
        response = requests.get(full_url, stream=True)
        response.raise_for_status()

        # Сохранение ZIP-файла
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        # Создание временной директории для распаковки
        temp_extract_dir = os.path.join(download_directory, 'temp_extract')
        os.makedirs(temp_extract_dir, exist_ok=True)

        # Распаковка ZIP-файла
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Найти HTML-файл
            html_files = [f for f in zip_ref.namelist() if f.lower().endswith('.html')]

            if not html_files:
                print("HTML-файл не найден в архиве")
                return None

            # Извлечь первый найденный HTML-файл
            html_filename = html_files[0]
            zip_ref.extract(html_filename, temp_extract_dir)

        # Полный путь к извлеченному HTML-файлу
        extracted_html_path = os.path.join(temp_extract_dir, html_filename)

        # Перезапись файла с правильной кодировкой
        with open(extracted_html_path, 'r', encoding='cp1251') as file:
            content = file.read()

        # Сохраняем текст в формате UTF-8
        safe_book_name = "".join(x for x in book_name if x.isalnum() or x in [' ', '_']).rstrip()
        safe_author_name = "".join(x for x in author_name if x.isalnum() or x in [' ', '_']).rstrip()

        new_filename = f"{safe_author_name}_{safe_book_name}.html".replace(' ', '_')
        final_html_path = os.path.join(download_directory, new_filename)

        with open(final_html_path, 'w', encoding='utf-8') as file:
            file.write(content)

        # Очистка временных файлов
        shutil.rmtree(temp_extract_dir)
        os.remove(zip_path)

        print(f"HTML-файл успешно скачан: {final_html_path}")
        return final_html_path

    except requests.RequestException as e:
        print(f"Ошибка при скачивании: {e}")
        return None
    except zipfile.BadZipFile:
        print("Ошибка: Некорректный ZIP-архив")
        return None
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")
        return None
    


def parse_book_structure(downloaded_html_path):
    with open(downloaded_html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    html_content = str(BeautifulSoup(html_content, "html.parser"))

    # Convert <br> to newlines for proper text handling
    html_content = html_content.replace('<br>', '\n')
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize the result dictionary
    book_structure = {}
    current_h1 = None
    current_h2 = None
    
    # Process each element
    for element in soup.find_all(['h1', 'h2', 'div']):
        if element.name == 'h1':
            current_h1 = element.get_text().strip()
            book_structure[current_h1] = {}
            current_h2 = None
        elif element.name == 'h2':
            if current_h1:
                current_h2 = element.get_text().strip()
                if current_h2 not in book_structure[current_h1]:
                    book_structure[current_h1][current_h2] = ""
        elif element.name == 'div':
            if current_h1 and current_h2:
                text = element.get_text().strip()
                book_structure[current_h1][current_h2] = text
    
    return book_structure


# Функция для подготовки данных
def flatten_structure_with_metadata(structure, parent_titles=None, chunk_size=1000, overlap=150):
    """Разворачивает структуру в список документов с метаданными заголовков, разбивая содержимое на чанки."""
    parent_titles = parent_titles or []
    documents = []

    def chunk_text(text, chunk_size, overlap):
        """Разбивает текст на чанки фиксированного размера с перекрытием."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap  # сдвиг с учетом перекрытия
        return chunks

    def process_structure(sub_structure, parent_titles):
        for title, content_or_sub in sub_structure.items():
            current_titles = parent_titles + [title]

            if isinstance(content_or_sub, dict):  # Если это вложенная структура
                process_structure(content_or_sub, current_titles)
            else:  # Если это текстовый контент
                content = content_or_sub.strip()
                if content:
                    headers_str = " > ".join(current_titles)
                    chunks = chunk_text(content, chunk_size, overlap)
                    for i, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "headers": headers_str,
                                    "chunk_index": i + 1,  # Номер чанка в разделе
                                    "total_chunks": len(chunks),  # Общее количество чанков
                                }
                            )
                        )

    process_structure(structure, parent_titles)
    return documents


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger()


def vectorize_documents(documents, collection_name="book_with_headers", persist_directory="./chroma_db/"):
    logger = setup_logging()
    
    total_docs = len(documents)
    logger.info(f"Начало векторизации {total_docs} документов")
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_emb_model(),
            persist_directory=persist_directory
        )
        
        # Добавляем документы с progress bar и логированием
        chunk_size = 10  # Размер пакета для обработки
        for i in tqdm(range(0, total_docs, chunk_size), desc="Векторизация документов"):
            chunk = documents[i:min(i + chunk_size, total_docs)]
            vectorstore.add_documents(chunk)
            
            docs_processed = min(i + chunk_size, total_docs)
            docs_remaining = max(total_docs - docs_processed, 0)
            
            # Выводим прогресс
            logger.info(f"Прогресс: {docs_processed}/{total_docs} документов | Осталось: {docs_remaining}")
        
        logger.info("Векторизация успешно завершена")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Ошибка при векторизации: {str(e)}")
        raise
    

def get_book_and_vector_store(book_name, author_name, download_directory="./books_library/"):

    download_book(book_name, author_name, download_directory=download_directory)
    download_directory = (
        download_directory + 
        author_name.replace(' ', '_') +
        '_' +
        book_name.replace(' ', '_') +
        '.html'
    )
    book_structure = parse_book_structure(download_directory)
    documents = flatten_structure_with_metadata(book_structure) 

    vectorstore = vectorize_documents(
        documents, 
        collection_name="book_with_headers", 
        persist_directory="./chroma_db/"
    )
    return vectorstore


def connect_to_existing_vectorstore(collection_name="book_with_headers", persist_directory="./chroma_db/"):
    """
    Подключается к существующей векторизированной базе данных Chroma.
    
    Args:
        collection_name (str): Название коллекции
        persist_directory (str): Путь к директории с базой данных
        
    Returns:
        Chroma: Объект векторного хранилища
    """
    logger = setup_logging()
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_emb_model(),
            persist_directory=persist_directory
        )
        logger.info(f"Успешно подключено к существующей базе данных: {collection_name}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Ошибка при подключении к базе данных: {str(e)}")
        raise