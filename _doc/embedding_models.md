# Embedding models [[1](#1)]

Los modelos de embedding transforman el lenguaje humano a un formato que las máquinas pueden entender y comparar con velocidad y precisión. Estos modelos recogen texto como input y producen un array de números de tamaño fijo, una huella numérica del significado semántico del texto. 

Los modelos de embedding permiten al sistema buscar documentos relevantes no solo por palabras clave relevantes, si no tambien por comprensión semántica.

### Conceptos clave

![key concepts](https://python.langchain.com/assets/images/embeddings_concept-975a9aaba52de05b457a1aeff9a7393a.png)

## Embedding en langchain

### Interfaz

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

### Medida de similaridad

Cada embedding es en esencia un conjunto de coordenadas, a menudo en un espacion multi-dimensional. En este espacio, cada punto (embedding) refleja el significado de su texto correspondiente, permitiendo asi realizar comparaciones entre diferentes piezas de texto.

Al reducir el texto a estas representaciones numéricas, podemos aplicar operaciones matemáticas para medir como se relacionan dos piezas de texto:

* **Similaridad coseno**: Mide el ángulo del coseno entre dos vectores.
* **Distancia Euclidea**: Mide la línea recta entre dos puntos.
* **Producto escalar**: Mide la proyección de un vector en otro.

## Modelos de embedding



## Referencias

<a name=1></a>
[1] Embedding Models, https://python.langchain.com/docs/concepts/embedding_models/
