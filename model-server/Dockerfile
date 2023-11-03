#Indiquem la versio de python
FROM python:3.8

#Directori en el que traballar el contenidor
WORKDIR /app

#Compiem els continguts del directori actual al contenidor
ADD . /app

#Instal·lem els requeriments
RUN pip install -r requirements-server.txt

#Executem la aplicacio flask
CMD ["flask", "--app", "main-server", "run", "--host=0.0.0.0"]
