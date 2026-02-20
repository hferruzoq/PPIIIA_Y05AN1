import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------
st.set_page_config(page_title="Evaluación IA de Postulantes", layout="centered")
st.title("🤖 Evaluación Inteligente de Postulantes")

# ----------------------------------
# CARGAR MODELO DE EMBEDDINGS
# ----------------------------------
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo_embeddings = cargar_modelo()

# ----------------------------------
# CARGAR PERFIL DEL PUESTO (PKL)
# ----------------------------------
with open("modelo/perfil_puesto.pkl", "rb") as f:
    perfil_puesto_embedding = pickle.load(f)

# ----------------------------------
# FORMULARIO DEL POSTULANTE
# ----------------------------------
st.header("📄 Formulario del Postulante")

with st.form("form_postulante"):
    nivel_academico = st.selectbox(
        "Nivel académico",
        ["Técnico", "Universitario", "Posgrado"]
    )

    carrera = st.text_input("Carrera profesional")

    experiencia_anios = st.number_input(
        "Años de experiencia en el puesto",
        min_value=0,
        max_value=40
    )

    descripcion_experiencia = st.text_area(
        "Describe tu experiencia laboral"
    )

    tecnologias = st.text_area(
        "Conocimientos tecnológicos",
        placeholder="Ejemplo: Python, SQL, Power BI"
    )

    habilidades = st.text_area(
        "Habilidades técnicas y blandas",
        placeholder="Ejemplo: análisis, trabajo en equipo"
    )

    certificaciones = st.text_area(
        "Certificaciones (opcional)"
    )

    enviar = st.form_submit_button("Evaluar Postulación")

# ----------------------------------
# PROCESAMIENTO
# ----------------------------------
if enviar:
    # Convertir formulario a texto semántico
    texto_postulante = f"""
    Nivel académico: {nivel_academico}.
    Carrera profesional: {carrera}.
    Experiencia laboral: {experiencia_anios} años.
    Descripción de experiencia: {descripcion_experiencia}.
    Tecnologías dominadas: {tecnologias}.
    Habilidades: {habilidades}.
    Certificaciones: {certificaciones}.
    """

    # Generar embedding del postulante
    embedding_postulante = modelo_embeddings.encode([texto_postulante])

    # Calcular similitud
    similitud = cosine_similarity(
        embedding_postulante,
        perfil_puesto_embedding
    )[0][0]

    porcentaje = round(similitud * 100, 2)

    # Umbral de decisión
    umbral = 0.70

    st.subheader("📊 Resultado de la Evaluación")
    st.write(f"**Similitud con el perfil del puesto:** {porcentaje}%")

    if similitud >= umbral:
        st.success("✅ Postulante APTO para el puesto")
    else:
        st.error("❌ Postulante NO APTO para el puesto")

    # ----------------------------------
    # GUARDAR DATOS EN CSV
    # ----------------------------------
    datos = {
        "nivel_academico": nivel_academico,
        "carrera": carrera,
        "experiencia_anios": experiencia_anios,
        "descripcion_experiencia": descripcion_experiencia,
        "tecnologias": tecnologias,
        "habilidades": habilidades,
        "certificaciones": certificaciones,
        "similitud": porcentaje
    }

    df = pd.DataFrame([datos])

    os.makedirs("data", exist_ok=True)

    archivo = "data/postulantes.csv"
    if os.path.exists(archivo):
        df.to_csv(archivo, mode="a", header=False, index=False)
    else:
        df.to_csv(archivo, index=False)

    st.info("📁 Postulación registrada correctamente")

