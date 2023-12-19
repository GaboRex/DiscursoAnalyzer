FROM python:3.11-slim
ENV PORT 8000

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN python -m spacy download es_core_news_md

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')"

COPY ./disAnalyzer /disAnalyzer

CMD uvicorn disAnalyzer.main:app --host 0.0.0.0 --port ${PORT}