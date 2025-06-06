FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY serve.sh /app/serve.sh
RUN chmod +x /app/serve.sh

ENTRYPOINT ["/app/serve.sh"]
CMD ["serve"]
