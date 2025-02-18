# Text Splitters [[1](#1)]

La separación o **split** de documentos es un paso esencial para muchas aplicaciones. Se trata de romper grandes textos en pequeños trozos o **chunks**.

![photo](https://python.langchain.com/assets/images/text_splitters-7961ccc13e05e2fd7f7f58048e082f47.png)

Hay muchas razones por las que separar textos:

* Manejo de documentos con longitudes no uniformes.
* Superación de los límites impuestos por los modelos.
* Mejora de la calidad de representación.
* Mejora de la presición de recuperación.
* Optimización de recursos computacionales.

## Split basado en longitud

La estrategia más intuitiva, con ella nos aseguramos que los chunks no sobrepasen un determinado límite. Tiene los siguientes beneficios:

* Implementación sencilla.
* Tamaños consistentes en los chunks.
* Fácilmente adaptable a los requerimientos de distintos modelos.

Hay diferentes tipos de maneras de separar textos basados en longitud:

* Basado en tokens: muy útil al trabajar con LLMs.
* Basado en caracteres: más consistente con diferentes tipos de texto.

Ejemplo de separación por tokens usando la clase *CharacterTextSplitter* [[2](#2)]:
```
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

## Split basado en texto

Aprovecha la forma jerárquica en la que organizado el lenguaje natural como estrategia para separar texto. Creando chunks que tienen coherencia semántica. La clase *RecursiveCharacterTextSplitter* [[3](#3)] implementa este concepto.

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
```

Los parámetros usados:

* **chunk_size**: Tamaño máximo del chunk determinado por la función de longitud.
* **chunk_overlap**: Solapamiento entre chunks, mitiga la pérdida de información.
* **length_function**: Función de longitud, para determinar el tamaño del chunk.
* **is_separator_regex**: lista de separadores. (Por defecto ``["\n\n", "\n", " ", ""]``)

## Separaciones basadas en la estructura del documento

Usados principalmente en documentos HTML, JSON, Markdown, etc. Se preserva la organización lógica del documento, el contexto entre chunks y es más efectivo en las tareas de búsqueda y sumarización.

Algunos ejemplos de los métodos usado para la separación de textos:

* **Markdown** [[4](#4)]: Separión basada en cabeceras (#, ##, ###)
* **JSON** [[5](#5)]: Separación por objeto o array de elementos.
* **HTML** [[6](#6)]: Separación por etiquetas.
* **Código** [[7](#7)]: Separación por funciones, clases o bloques de código.

## Separaciones basadas en la semántica

Este tipo de separación considera el contenido del texto, analiza la semántica. Esta técnica separa cuando hay cambios significativos en el significado del texto, por lo que crea chunks con más coherencia semántica beneficiando así la calidad de respuesta en búsquedas y sumarizaciones.

Los siguientes ejemplos están basados en el notebook de **Greg Kamradt: 5 Levels of Text Splitting** [[8](#8)]. 

El separador que se usa para esta tarea en langchain es *SemantycChunker* [[9](#9)] y sus parámetros son los siguientes: 
* embeddings (Embeddings)
* buffer_size (int)
* add_start_index (bool)
* breakpoint_threshold_type (Literal['percentile', 'standard_deviation', 'interquartile', 'gradient'])
* breakpoint_threshold_amount (float | None)
* number_of_chunks (int | None)
* sentence_split_regex (str)
* min_chunk_size (int | None)

Ejemplo con OpenAIEmbeddings:
```
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(OpenAIEmbeddings())
```

### Breakpoints

Límite por el que se determina la separación de los chunks. Esta clase posee varias formas de indicarlo.

#### Percentil

El valor por defecto para X es 95.0 y puede ser ajustado con el parámetro ``breakpoint_threshold_amount`` que espera un valor entre 0.0 y 100.0.

```
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)
```

#### Desviación estandar

Es valor por defecto para X es 3.0  y puede ser ajustado con el parámetro ``breakpoint_threshold_amount``.

```
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation"
)
```

#### Intercuartil

El rango intercuartil puede ser ajustado con el parámetro ``breakpoint_threshold_amount``, el valor por defecto es 1.5.

```
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="interquartile"
)
```

#### Gradiente

En este métodp, el gradiente es usado conjuntamente con el percentil, es útil cuando los chunks están altamente correlacionados. Similar a los demás, puede ser ajustado con el parámetro ``breakpoint_threshold_amount`` que espera un número entre 0.0 y 100.0, el valor por defecto es 95.0.

```
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="gradient"
)
```

## Referencias

<a name=1></a>
[1] Text Splitters, https://python.langchain.com/docs/concepts/text_splitters/

<a name=2></a>
[2] CharacterTextSplitter, https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.CharacterTextSplitter.html

<a name=3></a>
[3] How to recursively split text by characters, https://python.langchain.com/docs/how_to/recursive_text_splitter/

<a name=4></a>
[4] How to split Markdown by Headers, https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/

<a name=5></a>
[5] How to split JSON data, https://python.langchain.com/docs/how_to/recursive_json_splitter/

<a name=6></a>
[6] How to split HTML, https://python.langchain.com/docs/how_to/split_html/

<a name=7></a>
[7] How to split code, https://python.langchain.com/docs/how_to/code_splitter/

<a name=8></a>
[8] 5 Levels Of Text Splitting, https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

<a name=9></a>
[9] How to split text based on semantic similarity, https://python.langchain.com/docs/how_to/semantic-chunker/

