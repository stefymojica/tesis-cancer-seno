import streamlit as st
import pandas as pd
import sys
import os

# Añadir la carpeta src al path para poder importar módulos
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.data_loader import load_data, get_basic_stats

st.set_page_config(page_title="EDA - Exploración de Datos", page_icon="📊")

st.title("📊 Análisis Exploratorio de Datos (EDA)")

st.info("Esta página escanea automáticamente la carpeta `data/` en busca de archivos de CBIS-DDSM.")

# Intentar cargar datos del directiorio
from src.data_loader import load_cbis_ddsm_data, get_summary_stats
datasets = load_cbis_ddsm_data('data')

if datasets:
    selected_file = st.selectbox("Selecciona un dataset para explorar:", list(datasets.keys()))
    df = datasets[selected_file]
    
    col1, col2, col3 = st.columns(3)
    stats = get_summary_stats(df)
    col1.metric("Total de Casos", stats['total_cases'])
    col2.metric("Malignos", stats['malignant_count'])
    col3.metric("Benignos", stats['benign_count'])

    st.write("### Vista Previa de los Datos")
    st.dataframe(df.head())
    
    st.write("### Distribución de Patologías")
    if 'pathology' in df.columns:
        st.bar_chart(df['pathology'].value_counts())
    
    st.write("### Correlación de Variables")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        st.write(numeric_df.corr())
else:
    st.warning("No se encontraron archivos CSV de CBIS-DDSM en la carpeta `data/`.")
    st.markdown("""
    Para comenzar:
    1. Descarga los archivos de metadatos de [TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).
    2. Colócalos en la carpeta `data/`.
    3. O usa el archivo de simulación que he creado para ti.
    """)

# Mantener el cargador manual como respaldo
with st.expander("Subir un archivo manualmente"):
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv", key="manual_upload")
    if uploaded_file:
        manual_df = pd.read_csv(uploaded_file)
        st.dataframe(manual_df.head())
