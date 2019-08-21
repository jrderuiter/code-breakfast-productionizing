FROM python:3.7.1

ENV PYTHONDONTWRITEBYTECODE 1

ADD . /app
RUN pip install /app

EXPOSE 5000

ENTRYPOINT ["titanic"]
CMD ["serve", "--host", "0.0.0.0", "/model/model.pkl"]
