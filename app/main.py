import streamlit as st

st.set_page_config(
    page_title="Tesis: Cáncer de Seno",
    page_icon="🎗️",
    layout="wide"
)

st.title("🎗️ Análisis de Imágenes y Datos: Cáncer de Seno")
st.markdown("""
### Bienvenido a la plataforma de análisis para tu tesis.
Este tablero te permitirá visualizar datos, realizar un Análisis Exploratorio (EDA) y, próximamente, ejecutar modelos de detección de cáncer.

#### Estructura del Proyecto:
- **EDA**: Visualización y estadísticas descriptivas de los datos.
- **Detección**: Inferencia con modelos de Deep Learning (por implementar).
- **Documentación**: Guías y reportes técnicos.

Usa la barra lateral para navegar por las diferentes secciones.
""")

st.sidebar.success("Selecciona una sección arriba.")
