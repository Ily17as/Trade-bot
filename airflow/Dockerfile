FROM apache/airflow:2.10.0

# Работать нужно под пользователем airflow, а не root
USER airflow

# Кладём requirements внутрь образа
COPY requirements.txt /requirements.txt

# Устанавливаем Python-пакеты
RUN pip install --no-cache-dir -r /requirements.txt
