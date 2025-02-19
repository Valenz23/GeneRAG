# Embedding models [[1](#1)]

Los modelos de embedding transforman el lenguaje humano a un formato que las máquinas pueden entender y comparar con velocidad y precisión. Estos modelos recogen texto como input y producen un array de números de tamaño fijo, una huella numérica del significado semántico del texto. 

Los modelos de embedding permiten al sistema buscar documentos relevantes no solo por palabras clave relevantes, si no tambien por comprensión semántica.

### Conceptos clave

![key concepts](https://python.langchain.com/assets/images/embeddings_concept-975a9aaba52de05b457a1aeff9a7393a.png)

## 1. Embedding en langchain

### 1.2. Interfaz

Langchain ofrece una interfaz universal para trabajor con ellos, proveyendo métodos para operaciones comunes. Esta interfaz común simplifica la interacción con los distintos modelos de embedding a través de dos métodos centrales:

* **embed_documents**: Embebe multiples textos (documents).

    ```
    from langchain_ollama import OllamaEmbeddings
    embeddings_model = OllamaEmbeddings(model="llama3")
    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    len(embeddings), len(embeddings[0])
    (5, 1536)
    ```
* **embed_query**: Embebe un solo texto (query).

    ```
    embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    ```

### 1.3. Medida de similaridad

Cada embedding es en esencia un conjunto de coordenadas, a menudo en un espacion multi-dimensional. En este espacio, cada punto (embedding) refleja el significado de su texto correspondiente, permitiendo asi realizar comparaciones entre diferentes piezas de texto.

Al reducir el texto a estas representaciones numéricas, podemos aplicar operaciones matemáticas para medir como se relacionan dos piezas de texto:

* **Similaridad coseno**: Mide el ángulo del coseno entre dos vectores.
* **Distancia Euclidea**: Mide la línea recta entre dos puntos.
* **Producto escalar**: Mide la proyección de un vector en otro.

## 2. Modelos de embedding populares 

### 2.1. nomic-embed-text [[2](#2)]

El modelo nomic-embed-text-v1 es un modelo de embeddings de texto desarrollado por Nomic AI, diseñado para manejar contextos de hasta 8192 tokens. Este modelo destaca por ser completamente de código abierto, con datos y código de entrenamiento disponibles públicamente, lo que garantiza su reproducibilidad y auditabilidad. 

![nomic embed](https://www.nomic.ai/blog/nomic_beats_ada_wide.webp)

**Características principales**:

* Alto rendimiento en tareas de contexto corto y largo: nomic-embed-text-v1 supera a modelos propietarios como OpenAI's text-embedding-ada-002 y text-embedding-3-small en tareas que involucran tanto contextos cortos como largos.
* Entrenamiento transparente y reproducible: El modelo fue entrenado utilizando un conjunto de datos curado de aproximadamente 235 millones de pares de texto, siguiendo una metodología de aprendizaje contrastivo. Tanto el código de entrenamiento como los pesos del modelo están disponibles bajo la licencia Apache-2.
* Versatilidad en aplicaciones de PLN: nomic-embed-text-v1 es adecuado para una variedad de tareas de Procesamiento de Lenguaje Natural, incluyendo recuperación de información, búsqueda semántica, clasificación, clustering y visualización de datos.

**Limitaciones**:

* Idioma: El modelo está entrenado exclusivamente en inglés, por lo que su rendimiento en otros idiomas puede ser limitado.
* Tamaño del modelo: Con 137 millones de parámetros, puede requerir recursos computacionales significativos para su implementación en entornos con limitaciones de hardware. 

### 2.2. mxbai-embed-large [[3](#3)]

El modelo mxbai-embed-large-v1 es un modelo de embeddings en inglés de última generación desarrollado por Mixedbread. Destaca por su alto rendimiento en tareas de Procesamiento de Lenguaje Natural (NLP) y supera a modelos propietarios como el text-embedding-ada-002 de OpenAI.

| Model                         | Avg (56 datasets) | Classification (12 datasets) | Clustering (11 datasets) | PairClassification (3 datasets) | Reranking (4 datasets) | Retrieval (15 datasets) | STS (10 datasets) | Summarization (1 dataset) |
|-------------------------------|-------------------|-----------------------------|--------------------------|--------------------------------|------------------------|-------------------------|-------------------|--------------------------|
| mxbai-embed-large-v1          | 64.68             | 75.64                       | 46.71                    | 87.2                           | 60.11                  | 54.39                   | 85.00             | 32.71                    |
| bge-large-en-v1.5             | 64.23             | 75.97                       | 46.08                    | 87.12                          | 60.03                  | 54.29                   | 83.11             | 31.61                    |
| mxbai-embed-2d-large-v1       | 63.25             | 74.14                       | 46.07                    | 85.89                          | 58.94                  | 51.42                   | 84.9              | 31.55                    |
| nomic-embed-text-v1           | 62.39             | 74.12                       | 43.91                    | 85.15                          | 55.69                  | 52.81                   | 82.06             | 30.08                    |
| jina-embeddings-v2-base-en    | 60.38             | 73.45                       | 41.73                    | 85.38                          | 56.98                  | 47.87                   | 80.7              | 31.6                     |
| **Proprietary Models**        |                   |                             |                          |                                |                        |                         |                   |                          |
| OpenAI text-embedding-3-large | 64.58             | 75.45                       | 49.01                    | 85.72                          | 59.16                  | 55.44                   | 81.73             | 29.92                    |
| Cohere embed-english-v3.0     | 64.47             | 76.49                       | 47.43                    | 85.84                          | 58.01                  | 55.00                   | 82.62             | 30.18                    |
| OpenAI text-embedding-ada-002 | 60.99             | 70.93                       | 45.90                    | 84.89                          | 56.32                  | 49.25                   | 80.97             | 30.80                    |


**Características principales**:

* Entrenamiento extenso: El modelo fue entrenado con más de 700 millones de pares utilizando entrenamiento contrastivo y afinado con más de 30 millones de triplest de alta calidad mediante la función de pérdida AnglE. Esto le permite adaptarse a una amplia gama de temas y dominios, siendo ideal para aplicaciones del mundo real y casos de uso de Recuperación Aumentada por Generación (RAG).
* Eficiencia en almacenamiento y recuperación: Es adecuado para generar embeddings binarios, lo que permite ahorrar hasta 32 veces en almacenamiento y lograr una recuperación 40 veces más rápida, manteniendo más del 96% del rendimiento.
* Versatilidad en tareas NLP: El modelo ha demostrado un rendimiento destacado en el Massive Text Embedding Benchmark (MTEB), evaluando tareas como clasificación, clustering, clasificación de pares, re-ranking, recuperación, similitud textual semántica y resumen.

**Limitaciones**:

* Idioma: Está entrenado exclusivamente en inglés, por lo que su rendimiento en otros idiomas puede ser limitado.
* Longitud de secuencia: Se recomienda una longitud máxima de secuencia de 512 tokens; secuencias más largas pueden ser truncadas, lo que podría resultar en pérdida de información. 

### 2.3. snowflake-arctic-embed [[4](#4)] y snowflake-arctic-embed2 [[5](#5)]

El Snowflake Arctic Embed es una familia de modelos de embeddings de texto desarrollada por Snowflake, diseñada para proporcionar representaciones vectoriales de alta calidad para tareas de recuperación de información.

![snowflake1](https://publish-p57963-e462109.adobeaemcloud.com/adobe/dynamicmedia/deliver/dm-aid--1a17211a-dc96-49b7-bfcc-894a6e6e389e/frame-35-12.png?preferwebp=true&quality=85&width=960)

**Características principales**:

* Variedad de tamaños de modelos: La familia Arctic Embed incluye cinco modelos con diferentes cantidades de parámetros y dimensiones de embedding.
* Alto rendimiento en recuperación de información: El modelo más grande, snowflake-arctic-embed-l, ha demostrado un rendimiento superior en tareas de recuperación, con una puntuación NDCG@10 de 55.98 en MTEB.
* Entrenamiento y disponibilidad: Los modelos Arctic Embed fueron entrenados utilizando una variante del modelo bert-base-uncased, siguiendo una metodología de entrenamiento en múltiples etapas para optimizar su rendimiento en tareas de recuperación.


**Limitaciones**:

* Idioma: Aunque los modelos originales estaban entrenados principalmente en inglés, la versión más reciente, Arctic Embed 2.0, ha incorporado soporte multilingüe sin comprometer el rendimiento en inglés, ampliando su aplicabilidad a una gama más amplia de idiomas y casos de uso.

    ![snowflake2](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdo72tWVYWdUfUly2WGJtJ8rJSR6pfXf8_4If3J2Z4q6QHFTrw22_xmLp3_6p7JI5MZI5_3K7aJ7D4VerrHud7GH1XzNiFmQirirA6W0DeM4Ww-RKc8qvviFN9LAbiSi9Yu0DcI?key=W10CDsuqhacE_8Ljh1e0CcFQ)

* Requisitos de recursos: Los modelos más grandes, como snowflake-arctic-embed-l, pueden requerir recursos computacionales significativos para su implementación eficiente, lo que podría ser una consideración en entornos con limitaciones de hardware.

### 2.4. jina-embeddings-v2-base-es [[6](#6)]

El modelo jina-embeddings-v2-base-es es un modelo de embeddings de texto bilingüe diseñado para manejar entradas en inglés y español, admitiendo una longitud de secuencia de hasta 8192 tokens. Basado en la arquitectura BERT (JinaBERT), incorpora la variante bidireccional simétrica de ALiBi, lo que le permite procesar secuencias más largas de manera eficiente. Este modelo ha sido entrenado específicamente para aplicaciones monolingües y bilingües, garantizando un rendimiento óptimo incluso con entradas mixtas en inglés y español, sin sesgos.

![jina](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ZP2RVejCZovF3FDCg-Bx3A.png)

**Características principales**:

* Arquitectura avanzada: Utiliza JinaBERT con ALiBi bidireccional simétrico, permitiendo el procesamiento de secuencias largas de hasta 8192 tokens.
* Entrenamiento especializado: Diseñado para aplicaciones que requieren comprensión tanto monolingüe como bilingüe, manejando eficazmente entradas mixtas en inglés y español sin introducir sesgos.

**Limitaciones**:

* Enfoque en inglés y español: Aunque es altamente efectivo para estos dos idiomas, su rendimiento en otros idiomas no ha sido especificado y podría ser limitado.
* Requisitos de recursos: El procesamiento de secuencias largas puede demandar recursos computacionales significativos, lo que podría ser una consideración en entornos con limitaciones de hardware.

## Referencias

<a name=1></a>
[1] Embedding Models, https://python.langchain.com/docs/concepts/embedding_models/

<a name=2></a>
[2] Introducing Nomic Embed: A Truly Open Embedding Model, https://www.nomic.ai/blog/posts/nomic-embed-text-v1

<a name=3></a>
[3] Open Source Strikes Bread - New Fluffy Embedding Model, https://www.mixedbread.com/blog/mxbai-embed-large-v1

<a name=4></a>
[4] Snowflake Launches the World’s Best Practical Text-Embedding Model for Retrieval Use Cases, https://www.snowflake.com/en/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/

<a name=5></a>
[5] Snowflake’s Arctic Embed 2.0 Goes Multilingual: Empowering Global-Scale Retrieval with Inference Efficiency and High-Quality Retrieval, https://www.snowflake.com/en/engineering-blog/snowflake-arctic-embed-2-multilingual/?utm_source=chatgpt.com

<a name=6></a>
[6] jina/jina-embeddings-v2-base-es, https://ollama.com/jina/jina-embeddings-v2-base-es