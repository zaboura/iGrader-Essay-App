FROM python:3.6
 

WORKDIR /igraderapp

COPY /requirements.txt /igraderapp

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY . /igraderapp

#EXPOSE 5000

CMD ["python","app.py"]