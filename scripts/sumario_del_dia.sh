#!/bin/bash

# Obtener la fecha actual en formato YYYYMMDD
fecha=$(date +"%Y%m%d")

# Definir nombres de archivos
ficheroXML="../data/xml/$fecha.xml"
ficheroJSON="../data/json/$fecha.json"
ficheroPDF="../data/pdf/$fecha.pdf"

echo "📥 Iniciando descarga del sumario del día $(date)."

# Descargar los sumarios en XML y JSON
curl -s -L -X GET -H "Accept: application/xml" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$ficheroXML"
echo "✅ XML descargado: $ficheroXML"

curl -s -L -X GET -H "Accept: application/json" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$ficheroJSON"
echo "✅ JSON descargado: $ficheroJSON"

# Extraer la URL del PDF
urlPDF=$(sed -n 's/.*<url_pdf[^>]*>\(https[^<]*\)<\/url_pdf>.*/\1/p' "$ficheroXML" | head -n 1)

# Verificar si se encontró la URL 
if [[ -n "$urlPDF" ]]; then
    curl -s -L "$urlPDF" -o "$ficheroPDF"
    echo "✅ PDF descargado: $ficheroPDF"
else
    echo "⚠️ No se encontró un PDF."
fi

echo "🎉 Descarga completada."
