# Gunakan image dasar Apache Airflow
FROM apache/airflow:2.10.3

# Set working directory
WORKDIR /opt/airflow

# Salin requirements.txt dan instal dependencies tambahan
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Salin semua file proyek ke dalam container
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY config /opt/airflow/config

# Set environment variables jika diperlukan
ENV AIRFLOW_HOME=/opt/airflow

# Jalankan Airflow
CMD ["airflow", "webserver"]
