import spacy
import numpy as np
from sklearn.decomposition import PCA
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel

app = FastAPI(title="Detector de Tipos de Texto")

# Cargar el modelo de spaCy
nlp = spacy.load("es_core_news_md")

# Cargar los embeddings precalculados
with open("embeddings.npy", "rb") as f:
    embeddings = np.load(f)

# Realizar PCA para reducción de dimensionalidad
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

class TextAnalysis(BaseModel):
    text: str

@app.post("/analyze_text_type")
def analyze_text_type(text_data: TextAnalysis):
    text = text_data.text

    # Extraer el embedding del texto proporcionado
    text_embedding = extract_embedding(text, nlp).reshape(1, -1)

    # Aplicar PCA al embedding del texto para la reducción de dimensionalidad
    text_embedding_3d = pca.transform(text_embedding)

    # Calcular la similitud coseno entre el texto y los embeddings precalculados
    similarity_scores = cosine_similarity(text_embedding, embeddings)

    # Obtener los índices de los textos más similares
    indices_mas_similares = similarity_scores.argsort()[0][::-1]

    # Devolver los resultados
    results = {
        "text_type": "Discurso" if indices_mas_similares[0] < len(discursos) else "Canción" if indices_mas_similares[0] < len(discursos) + len(canciones) else "Noticia",
        "similarity_score": similarity_scores[0][indices_mas_similares[0]],
        "similar_texts": [{"index": indice, "similarity": similarity_scores[0][indice]} for indice in indices_mas_similares[1:5]],
    }

    return JSONResponse(content=results)
