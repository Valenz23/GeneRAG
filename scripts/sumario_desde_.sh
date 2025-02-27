#!/bin/bash

# Comprobar si se ha pasado una fecha de inicio como argumento
if [ -z "$1" ]; then
    echo "Por favor, proporciona una fecha de inicio en formato YYYYMMDD."
    exit 1
fi

# Fecha de inicio proporcionada por el usuario
fecha_inicio=$1

# Fecha de hoy (actual) en formato YYYYMMDD
fecha_hoy=$(date +"%Y%m%d")

# Validar que la fecha de inicio no sea posterior a la fecha de hoy
if [[ "$fecha_inicio" > "$fecha_hoy" ]]; then
    echo "La fecha de inicio no puede ser posterior a la fecha de hoy."
    exit 1
fi

# Convertir las fechas a formato YYYYMMDD
fecha_actual=$fecha_hoy
fecha=$fecha_inicio

# Iterar desde la fecha de inicio hasta hoy
while [[ "$fecha" -le "$fecha_actual" ]]; do

    # Comprobar si la fecha actual es un domingo (0 = domingo, 1 = lunes, etc.)
    dia_semana=$(date -d "$fecha" +%u)
    
    # Si es domingo, pasamos al siguiente día
    if [ "$dia_semana" -eq 7 ]; then
        fecha=$(date -d "$fecha + 1 day" +"%Y%m%d")
        continue
    fi

    sumario="../data/sumarios/$fecha.xml"
    
    # Descarga
    curl -s -L -X GET -H "Accept: application/xml" "https://www.boe.es/datosabiertos/api/boe/sumario/$fecha" -o "$sumario"
    echo "✅ XML descargado: $sumario"

    echo "🎉 Descarga completada para la fecha $fecha."

    # Incrementar la fecha en un día
    fecha=$(date -d "$fecha + 1 day" +"%Y%m%d")
done

echo "📚 Todos los sumarios han sido descargados desde $fecha_inicio hasta $fecha_hoy."
