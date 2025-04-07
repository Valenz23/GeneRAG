import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re

# Obtener la fecha actual
hoy = datetime.now()
fecha_hoy = hoy.strftime("%Y%m%d")  # Formato: YYYYMMDD

# Crear la ruta del archivo XML basado en la fecha
sumario = f"../data/sumarios/{fecha_hoy}.xml"

base_download_folder = "../data"
xml_folder = os.path.join(base_download_folder, "xml")
pdf_folder = os.path.join(base_download_folder, "pdf")
html_folder = os.path.join(base_download_folder, "web")

os.makedirs(xml_folder, exist_ok=True)
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(html_folder, exist_ok=True)

# Parsear el XML
tree = ET.parse(sumario)
root = tree.getroot()

# Función para descargar archivos
def descargar_archivo(url, carpeta, nombre_archivo):
    if not url or url == "N/A":
        return None
    
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    
    try:
        respuesta = requests.get(url, stream=True)
        respuesta.raise_for_status()  # Lanza una excepción si el status es 4xx o 5xx
        with open(ruta_completa, "wb") as archivo:
            for chunk in respuesta.iter_content(chunk_size=8192):
                archivo.write(chunk)
        return ruta_completa
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar {url}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error al guardar {nombre_archivo}: {e}")
        return None

# Función para descargar XML en su carpeta usando el identificador
def descargar_xml(url, identificador):
    return descargar_archivo(url, xml_folder, f"{identificador}.xml")

# Función para descargar PDF en su carpeta usando el identificador
def descargar_pdf(url, identificador):
    return descargar_archivo(url, pdf_folder, f"{identificador}.pdf")

# Función para guardar el enlace HTML en un archivo .txt usando el identificador
def guardar_enlace_html(url, identificador):
    if url and url != "N/A":
        ruta_completa = os.path.join(html_folder, f"{identificador}.txt")
        try:
            with open(ruta_completa, "w") as archivo:
                archivo.write(url)
            return ruta_completa
        except Exception as e:
            print(f"❌ Error al guardar el enlace HTML: {e}")
            return None
    return None

# Buscar y procesar los ítems que contienen "DANA"
for item in root.findall(".//item"):
    identificador = item.find("identificador").text if item.find("identificador") is not None else "N/A"
    titulo = item.find("titulo").text if item.find("titulo") is not None else "N/A"
    url_pdf = item.find("url_pdf").text if item.find("url_pdf") is not None else "N/A"
    url_xml = item.find("url_xml").text if item.find("url_xml") is not None else "N/A"
    url_html = item.find("url_html").text if item.find("url_html") is not None else "N/A"

    # Filtrar solo los que contienen "DANA"
    if re.search(r'\bDANA\b', titulo, re.IGNORECASE):
        print(f"📌 Identificador: {identificador}")
        print(f"📄 Título: {titulo}")
        print(f"🔗 Descargando XML: {url_xml}")
        print(f"🔗 Descargando PDF: {url_pdf}")
        print(f"🌐 Guardando enlace HTML: {url_html}")

        # Descargar XML y PDF, y guardar el enlace HTML
        archivo_xml = descargar_xml(url_xml, identificador)
        archivo_pdf = descargar_pdf(url_pdf, identificador)
        archivo_html = guardar_enlace_html(url_html, identificador)

        print(f"✅ XML guardado en: {archivo_xml if archivo_xml else '❌ Error en descarga'}")
        print(f"✅ PDF guardado en: {archivo_pdf if archivo_pdf else '❌ Error en descarga'}")
        print(f"✅ Enlace HTML guardado en: {archivo_html if archivo_html else '❌ Error en guardado'}")
        print("-" * 50)
