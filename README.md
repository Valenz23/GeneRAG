# ![GeneRAG](images/horizontal_logo.png)

<br>  

**👨‍💻 Autor:** Pablo Valenzuela Álvarez  
**📚 Director:** David Griol Barres

**📝 Calificación:** 9.3

## <img src="images/icon_logo.png" alt="logo" width="20"/> Resumen

En este Trabajo de Fin de Máster, hemos diseñado, desarrollado y evaluado un sistema de generación aumentada por recuperación o RAG. Aplicando diversas técnicas en la construcción, configuración y evaluación, hemos conseguido obtener un sistema capaz de responder preguntas con bastante precisión y de forma muy eficiente.

La base de un sistema RAG son los grandes modelos de lenguaje o LLM. Estos estan capacitados para responder preguntas generalistas, pero no funcionan de igual modo con datos con los que no han sido entrenados o privados. Para solucionar esta problemática, un sistema RAG necesita acceder a una base de conocimiento externa para construir una respuesta específica usando estos nuevos datos. Estas bases de conocimiento suelen ser bases de datos vectoriales que guardan la relación semántica entre los documentos, y los recuperan de forma rápida y eficiente ante una consulta de usuario.

En este proyecto hemos implementado nuestro propio sistema RAG junto a una interfaz web en la que podemos realizar consultas. También se ha elaborado un amplio conjunto de pruebas de rendimiento de las diferentes alternativas con las que podemos configurar un sistema RAG (modelos de lenguaje, embedding o estrategias de búsqueda) con el objetivo de averiguar la mejor combinación de elementos. Para ello, se han definido cuatro agentes críticos encargados de evaluar las respuestas generadas por el RAG.

Durante la realización de este proyecto hemos sopesado diferentes opciones para realizar nuestro sistema RAG. Hemos explorado y descartado plataformas con las que solíamos trabajar dado a que se han quedado obsoletas ante el auge de los grandes modelos de lenguaje actualmente. Librerías como LangChain (diseñada para trabajar con modelos de lenguaje), o Chroma (base de datos vectorial); han facilitado el desarrollo del proyecto gracias a su flexibilidad y adaptabilidad a diferentes modelos de lenguaje: desde modelos locales de Ollama, externos y gratuitos de Mistral o de pago con OpenAI.

Por último, la base de conocimiento usada para el proyecto contiene documentos del Boletín Oficial Español (BOE) con referencias a lo sucedido en la DANA de finales del año 2024. Cuando el proyecto comenzó, este evento estaba muy reciente y pensamos que era una buena idea la de proporcionar una plataforma de ayuda a los ciudadanos con dudas o preguntas sobre la DANA. Nuestro sistema RAG esta diseñado solo para este cometido, y a través de su interfaz web, responde con claridad y precisión cualquier duda siempre que tenga datos sólidos para ello.

## 🖼️ Capturas

<figure>
  <img src="images/interfaz_main.png" alt="main" width="450"/>
  <figcaption>Figura 1: Página principal de la aplicación</figcaption>
</figure>

<figure>
  <img src="images/interfaz-update.png" alt="update" width="450"/>
  <figcaption>Figura 2: Página de actualización de la base de datos</figcaption>
</figure>

## ⚙️ Tecnologías usadas  

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangChain](https://img.shields.io/badge/LangChain-ffffff?style=for-the-badge&logo=langchain&logoColor=green)
![Ollama](https://img.shields.io/badge/ollama-white?style=for-the-badge&logo=ollama&logoColor=black)
![MistralAI](https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF&style=for-the-badge)
![OpenAI](https://shields.io/badge/-OpenAI-ffffff?logo=openai&logoColor=37BC7D&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Jira](https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white)
