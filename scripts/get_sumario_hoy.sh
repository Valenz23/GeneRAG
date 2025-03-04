#!/bin/bash

# Obtener la fecha actual en formato YYYYMMDD
fecha=$(date +"%Y%m%d")

# Definir nombres del sumario
sumario="../data/sumarios/$fecha.xml"

echo "📥 Iniciando descarga del sumario del día $(date)."

# Descarga
curl -s -L -X GET -H "Accept: application/xml" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$sumario"
echo "✅ XML descargado: $sumario"

echo "🎉 Descarga completada."
