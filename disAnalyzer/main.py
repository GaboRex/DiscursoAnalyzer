from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from sklearn.decomposition import PCA
from embeddings_loader import load_embeddings, extract_embedding
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sqlmodel import Field, SQLModel, create_engine, Session
from docx import Document

class TextEntry(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    text: str
    text_type: str
    similarity_score: float

DATABASE_URL = "sqlite:///db.sqlite3"
engine = create_engine(DATABASE_URL)

SQLModel.metadata.create_all(engine)

app = FastAPI(title="Detector de Tipos de Texto")

texts, embeddings, nlp, discursos, canciones, noticias = load_embeddings()

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

@app.post("/analyze_text_type")
async def analyze_text_type(file: UploadFile = File(...)):
    if file.filename.endswith('.docx'):
        doc = Document(file.file)
        text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        text = await file.read()
        text = text.decode("utf-8")

    text_embedding = extract_embedding(text, nlp).reshape(1, -1)

    similarity_scores = cosine_similarity(text_embedding, embeddings)

    indices_mas_similares = similarity_scores.argsort()[0][::-1]

    if indices_mas_similares[0] < len(canciones):
        text_type = "Cancion"
    elif indices_mas_similares[0] < len(canciones) + len(discursos):
        text_type = "Discurso"
    else:
        text_type = "Noticia"

    results = {
        "text_type": text_type,
        "similarity_score": float(similarity_scores[0][indices_mas_similares[0]]),  
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