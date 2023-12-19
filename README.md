# IDiscurso 🗣️🥸

## Deploy nube☁️

        https://segundoparcial-topicsia-ygh66irtda-uc.a.run.app/docs

## Descripción 📚🔍
IDiscurso es una API diseñada para analizar archivos de texto y determinar si son discursos. Acepta archivos en formato .txt y .docx, y utiliza un modelo de lenguaje para extraer los embeddings del texto. Luego, compara estos embeddings con los embeddings precalculados de tres tipos de textos: discursos, noticias y canciones.

## Propósito 🎯
El propósito principal de IDiscurso es analizar archivos de texto y proporcionar información sobre si el texto es un discurso, así como detalles específicos sobre los textos más similares.

## Cómo funciona 🤖
IDiscursom utiliza modelos de procesamiento de lenguaje natural (NLP) para analizar archivos de texto. Incluye un endpoint para obtener si el texto es un discurso y un análisis detallado que incluye los textos más similares.

## Endpoint🔚
La API de IDiscursom tiene un endpoint principal:

* POST /**analyze_text_type**: Este endpoint recibe un archivo de texto (.txt o .docx) y devuelve si el texto es un discurso, junto con la puntuación de similitud y los textos similares.

## Uso 🖥️⌨️
Para utilizar la API de IDiscursom, simplemente realiza solicitudes POST con el archivo de texto que deseas analizar. La API procesará el texto y te devolverá si el texto es un discurso y un análisis detallado.

## Futuras mejoras 🚀
En el futuro, planeamos expandir la funcionalidad para analizar textos en diferentes idiomas y géneros.

También, se planea implementar funcionalidades específicas para analizar la estructura de los discursos.

## Contribuciones 🌀
Las contribuciones son bienvenidas. Si tienes alguna sugerencia o mejora, no dudes en abrir un problema o hacer un pull request.

## Autores:✒️
* **Gabriel Neme** -[GaboRex](https://github.com/GaboRex)
**56189**

* **Pablo Badani** -[Pabdro](https://github.com/Pabdro)
**55789**
