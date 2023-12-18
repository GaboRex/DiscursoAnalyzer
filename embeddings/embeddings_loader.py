import spacy
import numpy as np
from sklearn.decomposition import PCA

def extract_embedding(text: str, nlp: spacy.Language) -> np.ndarray:
    doc = nlp(text)
    return doc.vector

def load_embeddings():
    canciones = []
    discursos = []
    noticias = []

    with open("canciones.txt", encoding="utf-8") as f:
        text = f.read()
        canciones = text.split("===")

    with open("discursos.txt", encoding="utf-8") as f:
        text = f.read()
        discursos = text.split("===")

    with open("noticias.txt", encoding="utf-8") as f:
        text = f.read()
        noticias = text.split("===")

    nlp = spacy.load("es_core_news_md")

    texts = canciones + discursos + noticias
    embeddings = np.array([extract_embedding(text, nlp) for text in texts])

    return texts, embeddings, nlp, discursos, canciones, noticias