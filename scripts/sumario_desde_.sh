#!/bin/bash

# Comprobar si se ha pasado una fecha de inicio como argumento
if [ -z "$1" ]; then
    echo "Por favor, proporciona una fecha de inicio en formato YYYYMMDD."
    exit 1
fi

# Fecha de inicio proporcionada por el usuario (argumento)
fecha_inicio=$1

# Fecha de hoy (actual) en formato YYYYMMDD
fecha_hoy=$(date +"%Y%m%d")

# Validar que la fecha de inicio no sea posterior a la fecha de hoy
if [[ "$fecha_inicio" > "$fecha_hoy" ]]; then
    echo "La fecha de inicio no puede ser posterior a la fecha de hoy."
    exit 1
fi

# Convertir las fechas a formato YYYYMMDD (ya está en este formato, pero es para asegurar que estén bien)
fecha_actual=$fecha_hoy
fecha=$fecha_inicio

# Iterar desde la fecha de inicio hasta hoy (inclusive)
while [[ "$fecha" -le "$fecha_actual" ]]; do

    # Comprobar si la fecha actual es un domingo (0 = domingo, 1 = lunes, etc.)
    dia_semana=$(date -d "$fecha" +%u)
    
    # Si es domingo (día 7), pasamos al siguiente día
    if [ "$dia_semana" -eq 7 ]; then
        # echo "⚠️ El $fecha es domingo, no se descarga ningún archivo."
        fecha=$(date -d "$fecha + 1 day" +"%Y%m%d")
        continue
    fi

    # Definir los nombres de los archivos
    ficheroXML="../data/xml/$fecha.xml"
    ficheroJSON="../data/json/$fecha.json"
    ficheroPDF="../data/pdf/$fecha.pdf"
    
    # Descargar los sumarios en XML y JSON
    curl -s -L -X GET -H "Accept: application/xml" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$ficheroXML"
    curl -s -L -X GET -H "Accept: application/json" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$ficheroJSON"

    # Extraer la URL del PDF 
    urlPDF=$(sed -n 's/.*<url_pdf[^>]*>\(https[^<]*\)<\/url_pdf>.*/\1/p' "$ficheroXML" | head -n 1)

    # Verificar si se encontró la URL del PDF
    if [[ -n "$urlPDF" ]]; then
        # Descargar el PDF usando la URL
        curl -s -L "$urlPDF" -o "$ficheroPDF"
    else
        echo "⚠️ No se encontró un PDF."
    fi

    echo "🎉 Descarga completada para la fecha $fecha."

    # Incrementar la fecha en un día
    fecha=$(date -d "$fecha + 1 day" +"%Y%m%d")
done

echo "📚 Todos los archivos han sido descargados desde $fecha_inicio hasta $fecha_hoy."
