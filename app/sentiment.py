from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time

model_es = AutoModelForSequenceClassification.from_pretrained('karina-aquino/spanish-sentiment-model')
tokenizer_es = AutoTokenizer.from_pretrained('karina-aquino/spanish-sentiment-model')
sentiment_analyzer_es = pipeline('sentiment-analysis', model=model_es, tokenizer=tokenizer_es)

model_en = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_en = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_analyzer_en = pipeline('sentiment-analysis', model=model_en, tokenizer=tokenizer_en)

def analyze_sentiment(text, language='es'):
    start_time = time.time()

    model, tokenizer, analyzer = (model_es, tokenizer_es, sentiment_analyzer_es) if language == 'es' else (model_en, tokenizer_en, sentiment_analyzer_en)

    result = analyzer(text)
    score_normalized = (result[0]['score'] - 0.5) * 2

    threshold_negative = -0.5
    threshold_positive = 0.5

    if score_normalized < threshold_negative:
        sentiment_label = 'muy negativo'
    elif threshold_negative <= score_normalized < 0:
        sentiment_label = 'negativo'
    elif 0 <= score_normalized < threshold_positive:
        sentiment_label = 'neutral'
    elif threshold_positive <= score_normalized <= 1:
        sentiment_label = 'positivo'
    else:
        sentiment_label = 'muy positivo'

    execution_time = time.time() - start_time

    return {
        "text": text,
        "score": score_normalized,
        "sentiment": sentiment_label,
        "execution_time": execution_time
    }