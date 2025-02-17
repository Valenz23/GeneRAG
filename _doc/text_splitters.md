# Text Splitters

Una vez cargados los documentos, hay que descomponerlos en chunks (trozos) más pequeños. Langchain ofrece un conjunto de transformadores para hacer fácil esta tarea, permitiendo: separar, combinar, filtrar, etc.

A grandes rasgos, el separador de texto funciona de la siguiente manera:
1. Separa el texto en pequeños chunks con semejanza semántica (a menudo frases).
2. Combina los chunks pequeños en otros más grandes (medida controlada con alguna función).
3. Cuando se alcanza la medida deseada, se genera una pieza de texto de ese chunk y se empieza a crear el siguiente chunk con un poco se superposición (para mantener el contexto entre los chunks).

Hay dos maneras de personalizar nuestro text splitter:
* Definir como se separa el texto.
* Definir el tamaño del chunk.

## Tipos de Text Splitters

| Nombre | Clases | Separa en | Añade metadatos | Descripción |
|---|---|---|---|---|
| Recursive | [RecursiceCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/), [RecursiveJsonSplitter](https://python.langchain.com/docs/how_to/recursive_json_splitter/) | Lista de caracteres definidos por el usuario | ❌ | Separa el texto de forma recursiva, intentando mantener relacionadas las partes de texto cercanas. Esta es la forma recomendada para dividir texto.|
| HTML | [HTMLHeaderTextSplitter](https://python.langchain.com/docs/how_to/split_html/#using-htmlheadertextsplitter), [HTMLSectionSplitter](https://python.langchain.com/docs/how_to/split_html/#using-htmlsectionsplitter). | Caracteres HTML | ✅ | Divide el texto en caracteres HTML. Añade información sobre el chunk del caracter. |
|Markdown|[MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)|Caracteres Markdown|✅|Separa el texto en caracteres Markdown. Añade información sobre el chunk del caracter.|
|Código|[Lenguajes](https://python.langchain.com/docs/how_to/code_splitter/)|Caracteres de código en diferentes lenguajes|❌|Separa el texto basandose en los caracteres de 15 lenguajes de programación. Añade información sobre el chunk del caracter.|
| Tokens | [Clases](https://python.langchain.com/docs/how_to/split_by_token/) | Tokens | ❌ | Separa en tokens, hay diferentes maneras de medirlos. |
| Caracter | [CharacterTextSplitter](https://python.langchain.com/docs/how_to/character_text_splitter/) | Caracter definido por el usuario | ❌ | Separa el texto en basandose en el caracter definido por el usuario. |
| [<span style="color:lime">Experimental</span>] Semantic Chunker | [SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/) | Frases | ❌ | Separa en frases, luego las combina si son parecidas semánticamente. |
| AI21 Semantic Text Splitter | [AI21SemanticTextSplitter](https://python.langchain.com/docs/integrations/document_transformers/ai21_semantic_text_splitter/) | Tópicos | ✅ | Identifica tópicos que formen trozos coherentes de texto y separa a partir de ahí |