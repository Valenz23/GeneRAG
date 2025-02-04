FROM python:3.10

# Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos si lo tienes
COPY requirements.txt ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY app/ ./

# Exponer el puerto de Streamlit (por defecto 8501)
EXPOSE 8501

# Comando por defecto para ejecutar Streamlit
CMD ["streamlit", "run", "app.py"]