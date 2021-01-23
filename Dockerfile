FROM python:3.6
 

# RUN conda create -n igrader-env python=3.6
# RUN echo "source activate env" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH


WORKDIR /igraderapp
COPY /requirements.txt /igraderapp
RUN pip install --upgrade pip
#RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY . /igraderapp

EXPOSE 5000

CMD ["python","app.py"]