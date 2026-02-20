import streamlit as st
import pandas as pd
import pdfplumber
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(
    page_title="Evaluación Inteligente de CVs",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Sistema Inteligente de Evaluación de Postulantes")

modelo = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- FUNCIONES ----------------
def extraer_texto_pdf(archivo_pdf):
    texto = ""
    with pdfplumber.open(archivo_pdf) as pdf:
        for pagina in pdf.pages:
            if pagina.extract_text():
                texto += pagina.extract_text()
    return texto

def generar_texto_puesto(df):
    fila = df.iloc[0]
    texto = f"""
    Puesto: {fila['puesto']}.
    Habilidades requeridas: {fila['habilidades']}.
    Experiencia mínima: {fila['experiencia']} años.
    Nivel académico: {fila['nivel']}.
    Tecnologías clave: {fila['tecnologias']}.
    """
    return texto

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 Perfil del Puesto")

puesto = st.sidebar.text_input("Puesto")
habilidades = st.sidebar.text_area("Habilidades requeridas (separadas por coma)")
experiencia = st.sidebar.number_input("Experiencia mínima (años)", min_value=0, max_value=30)
nivel = st.sidebar.selectbox("Nivel académico", ["técnico", "universitario", "posgrado"])
tecnologias = st.sidebar.text_area("Tecnologías clave")

if st.sidebar.button("💾 Guardar Perfil del Puesto"):
    data = {
        "puesto": [puesto],
        "habilidades": [habilidades],
        "experiencia": [experiencia],
        "nivel": [nivel],
        "tecnologias": [tecnologias]
    }

    df_puesto = pd.DataFrame(data)
    df_puesto.to_csv("perfil_puesto.csv", index=False)

    texto_puesto = generar_texto_puesto(df_puesto)
    embedding_puesto = modelo.encode(texto_puesto)

    joblib.dump(embedding_puesto, "perfil_puesto.pkl")

    st.sidebar.success("Perfil guardado y procesado correctamente")

# ---------------- CUERPO PRINCIPAL ----------------
st.header("📎 Evaluación del CV del Postulante")

archivo_cv = st.file_uploader("Sube el CV del postulante (PDF)", type=["pdf"])

if archivo_cv and os.path.exists("perfil_puesto.pkl"):

    with st.spinner("Analizando CV..."):
        texto_cv = extraer_texto_pdf(archivo_cv)
        embedding_cv = modelo.encode(texto_cv)

        embedding_puesto = joblib.load("perfil_puesto.pkl")

        similitud = cosine_similarity(
            [embedding_puesto],
            [embedding_cv]
        )[0][0]

        porcentaje = round(similitud * 100, 2)

    st.subheader("📊 Resultado de la Evaluación")
    st.metric("Nivel de coincidencia", f"{porcentaje} %")

    if similitud >= 0.75:
        st.success("✅ POSTULANTE APTO PARA EL PUESTO")
    else:
        st.error("❌ POSTULANTE NO APTO PARA EL PUESTO")

    st.info(
        "La evaluación se basa en la similitud semántica entre el perfil del puesto "
        "y el contenido del CV utilizando modelos de lenguaje preentrenados."
    )

elif archivo_cv:
    st.warning("Primero debe registrar el perfil del puesto.")