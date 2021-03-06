FROM python:3.6
 

WORKDIR /igraderapp


COPY /requirements.txt /igraderapp

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY . /igraderapp


CMD ["python","app.py"]

EXPOSE 80