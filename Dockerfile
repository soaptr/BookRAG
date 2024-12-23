# Используем официальный Python-образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY app.py /app/
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Указываем команду запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]