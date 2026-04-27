# Pipeline Google Colab: viroma oscuro de Culicidae neotropicales

Archivo principal:

`pipeline_viroma_oscuro_google_colab.py`

## Proposito

Este pipeline organiza los insumos de datos para el articulo de reflexion sobre el viroma oscuro de Culicidae neotropicales. No escribe el manuscrito; deja listos los productos que despues puedes usar con Claude:

- Dataset curado de estudios Colombia-Brasil.
- Tabla 1: comparativa de herramientas bioinformaticas.
- Tabla 2: estudios/ecosistemas y brechas de deteccion.
- Figura 1: mapa bibliometrico y barras de evidencia.
- Figura 2: pipeline convencional vs pipeline aumentado con IA.
- Figura 3: comparativo regional del viroma oscuro.
- Brief tecnico en Markdown para alimentar la escritura del manuscrito.

## Como usarlo en Google Colab

1. Crea un notebook nuevo en Colab.
2. Abre `pipeline_viroma_oscuro_google_colab.py`.
3. Copia los bloques marcados con `# %%` y `# %% [markdown]` en celdas separadas.
4. Ejecuta desde la celda 1 hasta la 12.
5. Descarga la carpeta `outputs` o el ZIP generado al final.

## Flujo recomendado

Ejecuta primero el flujo ligero:

1. Configuracion y dependencias.
2. Parametros editoriales.
3. Dataset curado.
4. Exportacion de tablas.
5. Figuras 1-3.
6. Brief tecnico para Claude.

El modulo SRA/DIAMOND es opcional y pesado. Solo conviene ejecutarlo si necesitas reprocesar lecturas publicas o tus datos propios desde FASTQ/contigs.

## Regla metodologica clave

El pipeline separa tres niveles de evidencia:

- `exact`: porcentaje publicado o calculado directamente desde cifras publicadas.
- `partial`: evidencia parcial, util para contexto pero no como porcentaje central.
- `qualitative`: patron descrito en el articulo fuente sin porcentaje publicable.

Las barras rayadas en las figuras representan evidencia cualitativa. No deben interpretarse como porcentajes reales.

## Salidas esperadas

En Colab se creara:

`/content/viroma_oscuro_colab/outputs/`

Con subcarpetas:

- `figuras/`
- `tablas/`
- `scope_articulo.json`
- `brief_tecnico_para_claude.md`

