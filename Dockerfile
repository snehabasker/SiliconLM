FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends iverilog \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY . .
RUN python rag/datasheet_rag.py build --docs rag/sample_docs --index rag/index

EXPOSE 7860
CMD ["python", "deploy/app.py"]
