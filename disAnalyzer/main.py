from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from embeddings_loader import load_embeddings, extract_embedding
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sqlmodel import Field, SQLModel, create_engine, Session

# Define el modelo de datos para la tabla en la base de datos
class TextEntry(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    text: str
    text_type: str
    similarity_score: float

# Configura la base de datos SQLite
DATABASE_URL = "sqlite:///db.sqlite3"
engine = create_engine(DATABASE_URL)

# Crea las tablas en la base de datos
SQLModel.metadata.create_all(engine)

app = FastAPI(title="Detector de Tipos de Texto")

texts, embeddings, nlp, discursos, canciones, noticias = load_embeddings()

# Realizar PCA para reducción de dimensionalidad
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

@app.post("/analyze_text_type")
async def analyze_text_type(file: UploadFile = File(...)):
    text = await file.read()
    text = text.decode("utf-8")

    # Extraer el embedding del texto proporcionado
    text_embedding = extract_embedding(text, nlp).reshape(1, -1)

    # Calcular la similitud coseno entre el texto y los embeddings precalculados
    similarity_scores = cosine_similarity(text_embedding, embeddings)

    # Obtener los índices de los textos más similares
    indices_mas_similares = similarity_scores.argsort()[0][::-1]

    # Determinar el tipo de texto más similar
    if indices_mas_similares[0] < len(canciones):
        text_type = "Cancion"
    elif indices_mas_similares[0] < len(canciones) + len(discursos):
        text_type = "Discurso"
    else:
        text_type = "Noticia"

    # Devolver los resultados
    results = {
        "text_type": text_type,
        "similarity_score": float(similarity_scores[0][indices_mas_similares[0]]),  # Convertir a float64
        "similar_texts": [{"index": int(indice), "similarity": float(similarity_scores[0][indice])} for indice in indices_mas_similares[1:5]],  # Convertir a float64
    }
    db_text = TextEntry(
        text=text,
        text_type=text_type,
        similarity_score=float(similarity_scores[0][indices_mas_similares[0]]),
    )
    with Session(engine) as session:
        session.add(db_text)
        session.commit()

    return JSONResponse(content=results)