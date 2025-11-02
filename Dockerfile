# Usa la imagen base de Python (de preferencia la versión que estés utilizando en tu proyecto)
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de tu proyecto al contenedor
COPY . /app

# Instala las dependencias del proyecto desde el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8888 si deseas acceder a Jupyter Notebooks o Kedro-Viz
EXPOSE 8888

# Establece el comando predeterminado para ejecutar Kedro (correr el pipeline por defecto)
CMD ["kedro", "run"]
