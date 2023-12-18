#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
print(len(canciones))
print(len(discursos))
print(len(noticias))


# In[3]:


import spacy
import numpy as np


# In[4]:


def extract_embedding(text: str, nlp: spacy.Language) -> np.ndarray:
    doc = nlp(text)
    return doc.vector


# In[5]:


nlp = nlp = spacy.load("es_core_news_md")

texts = canciones + discursos + noticias
embeddings = np.array([extract_embedding(text, nlp) for text in texts])
embeddings[:4, :]


# In[6]:


len(embeddings[1])


# In[7]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[8]:


pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)
embeddings_3d.shape


# In[9]:


fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    embeddings_3d[:, 0],
    embeddings_3d[:, 1],
    embeddings_3d[:, 2],
)


# In[11]:


color = [2 for i in discursos] + [3 for i in noticias] + [4 for i in canciones]
color


# In[12]:


pca2 = PCA(n_components=2)
embeddings_2d = pca2.fit_transform(embeddings)


# In[14]:


plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color, s=10)  # ajusta el valor de s según sea necesario
# ajusta el valor de s según sea necesario


# In[15]:


plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color, s=10)  # ajusta el valor de s según sea necesario

# Etiquetar cada punto con su índice
for i, txt in enumerate(texts):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i))

# Mostrar el gráfico
plt.show()


# In[16]:


from sklearn.metrics.pairwise import cosine_similarity

# Calcular la similitud de coseno entre los vectores de incrustación
similarity_matrix = cosine_similarity(embeddings)

# Imprimir la matriz de similitud
print(similarity_matrix)


# In[17]:


indice_texto_interes = 2  

similarity_scores = cosine_similarity(embeddings[indice_texto_interes].reshape(1, -1), embeddings)

indices_mas_similares = similarity_scores.argsort()[0][::-1]

print("Textos más similares al texto de interés:")
for indice in indices_mas_similares[1:]:
    print(f"Índice: {indice}, Similitud: {similarity_scores[0][indice]}")


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

indice_texto_interes = 2

similarity_scores = cosine_similarity(embeddings[indice_texto_interes].reshape(1, -1), embeddings)

indices_mas_similares = similarity_scores.argsort()[0][::-1]

print("\nTextos más similares al texto de interés:")
for indice in indices_mas_similares[1:]:
    categoria_texto_similar = None

    if indice <= 9:
        categoria_texto_similar = "discursos"
    elif indice <= 19 and indice >= 10  :
        categoria_texto_similar = "noticias"
    else:
        categoria_texto_similar = "canciones"

    print(f"Índice: {indice}, Similitud: {similarity_scores[0][indice]}, Categoría: {categoria_texto_similar}")


# In[22]:


from sklearn.metrics.pairwise import euclidean_distances

centroid_farandula = np.mean(embeddings[:len(discursos)], axis=0)
centroid_politica = np.mean(embeddings[len(discursos):len(discursos) + len(noticias)], axis=0)
centroid_deportes = np.mean(embeddings[len(discursos) + len(noticias):], axis=0)

centroids = [centroid_farandula, centroid_politica, centroid_deportes]
categories = ['farandula', 'politica', 'deportes']

distance_matrix = euclidean_distances(centroids, centroids)

print("Matriz de Distancias entre Categorías:")
print(distance_matrix)


# In[23]:


from scipy.spatial.distance import cdist
category_distances = cdist(embeddings, embeddings, metric='cosine')

average_distances = {
    "discursos-noticias": np.mean(category_distances[:len(discursos), len(discursos):len(discursos)+len(noticias)]),
    "discursos-canciones": np.mean(category_distances[:len(discursos), -len(canciones):]),
    "noticias-canciones": np.mean(category_distances[-len(noticias):, -len(canciones):])
}

print("\nDistancia promedio entre categorías:")
for key, value in average_distances.items():
    print(f"{key}: {value}")

