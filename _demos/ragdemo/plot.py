import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_graph_retriever(results, edges, filename="graph_retriever_output.png"):
    """
    Genera un grafo a partir de los documentos recuperados por GraphRetriever y lo guarda como imagen.
    
    results: lista de documentos recuperados.
    edges: lista de tuplas que definen las relaciones entre los nodos.
    filename: nombre del archivo donde se guardará el gráfico.
    """
    G = nx.Graph()
    node_labels = {}
    metadata_labels = {}

    # Crear los nodos y las etiquetas
    for doc in results:
        doc_id = doc.metadata.get("id", "Unknown ID")
        source = doc.metadata.get("source", "Unknown Source")
        author = doc.metadata.get("author", "Unknown Author")
        title = doc.metadata.get("title", "No Title")

        G.add_node(doc_id)
        node_labels[doc_id] = doc_id  # Etiqueta solo el ID
        metadata_labels[doc_id] = f"📄 {title}\n✍️ {author}\n📂 {source}"

    # Añadir las relaciones entre los nodos
    for doc in results:
        doc_id = doc.metadata.get("id", "Unknown ID")
        for related_doc in results:
            if doc != related_doc:
                for edge_type in edges:
                    key = edge_type[0]
                    if doc.metadata.get(key) == related_doc.metadata.get(key):
                        G.add_edge(doc_id, related_doc.metadata.get("id", "Unknown ID"))

    # Crear la figura y posicionar los nodos
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)

    # Dibujar el grafo
    nx.draw(G, pos, with_labels=False, node_color="lightblue", edge_color="gray",
            node_size=3000, font_size=10, font_weight="bold", alpha=0.6)

    # Ajustar las posiciones de las etiquetas para que no tapen los nombres
    for node, (x, y) in pos.items():
        # Coloca la etiqueta del nodo un poco por encima
        plt.text(x, y + 0.1, node_labels[node], fontsize=10, ha='center', fontweight='bold')

        # Coloca la metadata un poco por debajo
        plt.text(x, y - 0.1, metadata_labels[node], fontsize=8, ha='center', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # Título del gráfico
    plt.title("GraphRetriever Visualization")

    # Guardar el gráfico como una imagen
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico guardado como: {os.path.abspath(filename)}")

    # Cerrar la figura para liberar memoria
    plt.close()
