#Versio de python a utilitzar
FROM python:3.8

#Directori en el que treballar el contenidor
WORKDIR /app

#Copiem els arxius del nostre directori actual al del contenidor
ADD . /app

#Instal·lem els requeriments necessaris
RUN pip install -r requirements-train.txt

#Executem el script que entrena i guarda el model
CMD ["python", "main-train.py"]
