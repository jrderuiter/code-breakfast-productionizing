FROM python:3.7.1

ENV PYTHONDONTWRITEBYTECODE 1

ADD . /app
RUN pip install /app

EXPOSE 5000
CMD ["titanic", "serve", "--host", "0.0.0.0", "/model/model.pkl"]
