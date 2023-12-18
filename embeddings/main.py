from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from embeddings_loader import load_embeddings, extract_embedding
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Detector de Tipos de Texto")

texts, embeddings, nlp, discursos, canciones, noticias = load_embeddings()


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
    similarity_scores = cosine_similarity(text_embedding, embeddings).astype('float64')

    # Obtener los índices de los textos más similares
    indices_mas_similares = similarity_scores.argsort()[0][::-1]

    # Devolver los resultados
# Devolver los resultados
    results = {
        "text_type": "Discurso" if indices_mas_similares[0] < len(discursos) else "Canción" if indices_mas_similares[0] < len(discursos) + len(canciones) else "Noticia",
        "similarity_score": float(similarity_scores[0][indices_mas_similares[0]]),  # Convertir a float64
        "similar_texts": [{"index": int(indice), "similarity": float(similarity_scores[0][indice])} for indice in indices_mas_similares[1:5]],  # Convertir a float64
    }


    return JSONResponse(content=results)
