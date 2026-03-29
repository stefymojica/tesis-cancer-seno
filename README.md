# Tesis: Cáncer de Seno - Análisis y Detección

Este proyecto utiliza **Streamlit** para la visualización de datos y resultados de modelos de Deep Learning.

## Estructura del Proyecto

- `data/`: Datasets originales y procesados.
- `notebooks/`: Análisis exploratorio y pruebas de modelos (Jupyter Notebooks).
- `src/`: Código fuente modular (funciones de carga, preprocesamiento, modelos).
- `app/`: Aplicación de Streamlit.
    - `main.py`: Página de inicio.
    - `pages/`: Secciones adicionales (EDA, Inferencia, etc.).
- `docs/`: Documentación técnica y académica.

## Cómo empezar

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app/main.py
   ```