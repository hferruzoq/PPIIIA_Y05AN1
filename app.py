import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- CONFIGURACIÓN ----------------

st.set_page_config(
    page_title="Evaluación Inteligente de Postulantes",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Sistema Inteligente de Evaluación de Postulantes")

# ---------------- RUTAS SEGURAS (para GitHub y Streamlit Cloud) ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(BASE_DIR, "modelo")
RUTA_DATA = os.path.join(BASE_DIR, "data")

ARCHIVO_PUESTO_CSV = os.path.join(RUTA_MODELO, "perfil_puesto.csv")
ARCHIVO_PUESTO_PKL = os.path.join(RUTA_MODELO, "perfil_puesto.pkl")
ARCHIVO_POSTULANTES = os.path.join(RUTA_DATA, "postulantes.csv")

# Crear carpetas si no existen
os.makedirs(RUTA_MODELO, exist_ok=True)
os.makedirs(RUTA_DATA, exist_ok=True)

# ---------------- CARGAR MODELO NLP ----------------

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

modelo = cargar_modelo()

# ---------------- FUNCIONES ----------------

def generar_texto_puesto(df):
    fila = df.iloc[0]

    texto = f"""
    Puesto: {fila['puesto']}.
    Habilidades requeridas: {fila['habilidades']}.
    Experiencia mínima: {fila['experiencia']} años.
    Nivel académico requerido: {fila['nivel']}.
    Tecnologías clave: {fila['tecnologias']}.
    """

    return texto


def generar_texto_postulante(datos):
    texto = f"""
    Nivel académico: {datos['nivel']}.
    Carrera profesional: {datos['carrera']}.
    Experiencia laboral: {datos['experiencia']} años.
    Descripción de experiencia: {datos['descripcion']}.
    Tecnologías: {datos['tecnologias']}.
    Habilidades: {datos['habilidades']}.
    Certificaciones: {datos['certificaciones']}.
    """

    return texto


# ---------------- SIDEBAR PERFIL DEL PUESTO ----------------

st.sidebar.header("📌 Perfil del Puesto")

puesto = st.sidebar.text_input("Puesto")

habilidades_puesto = st.sidebar.text_area(
    "Habilidades requeridas"
)

experiencia_puesto = st.sidebar.number_input(
    "Experiencia mínima (años)",
    min_value=0,
    max_value=50
)

nivel_puesto = st.sidebar.selectbox(
    "Nivel académico requerido",
    ["técnico", "universitario", "posgrado"]
)

tecnologias_puesto = st.sidebar.text_area(
    "Tecnologías clave"
)

# Guardar perfil del puesto

if st.sidebar.button("💾 Guardar Perfil del Puesto"):

    data = {
        "puesto": [puesto],
        "habilidades": [habilidades_puesto],
        "experiencia": [experiencia_puesto],
        "nivel": [nivel_puesto],
        "tecnologias": [tecnologias_puesto]
    }

    df_puesto = pd.DataFrame(data)

    df_puesto.to_csv(ARCHIVO_PUESTO_CSV, index=False)

    texto_puesto = generar_texto_puesto(df_puesto)

    embedding_puesto = modelo.encode(texto_puesto)

    joblib.dump(embedding_puesto, ARCHIVO_PUESTO_PKL)

    st.sidebar.success("✅ Perfil del puesto guardado correctamente")


# ---------------- FORMULARIO POSTULANTE ----------------

st.header("📎 Formulario del Postulante")

with st.form("form_postulante"):

    nivel_postulante = st.selectbox(
        "Nivel académico",
        ["técnico", "universitario", "posgrado"]
    )

    carrera_postulante = st.text_input(
        "Carrera profesional"
    )

    experiencia_postulante = st.number_input(
        "Años de experiencia",
        min_value=0,
        max_value=50
    )

    descripcion_postulante = st.text_area(
        "Descripción de experiencia laboral"
    )

    tecnologias_postulante = st.text_area(
        "Tecnologías que domina"
    )

    habilidades_postulante = st.text_area(
        "Habilidades"
    )

    certificaciones_postulante = st.text_area(
        "Certificaciones"
    )

    boton_evaluar = st.form_submit_button("🔍 Evaluar Postulante")


# ---------------- EVALUACIÓN ----------------

if boton_evaluar:

    if not os.path.exists(ARCHIVO_PUESTO_PKL):

        st.warning("⚠ Primero debe registrar el perfil del puesto")

    else:

        datos_postulante = {
            "nivel": nivel_postulante,
            "carrera": carrera_postulante,
            "experiencia": experiencia_postulante,
            "descripcion": descripcion_postulante,
            "tecnologias": tecnologias_postulante,
            "habilidades": habilidades_postulante,
            "certificaciones": certificaciones_postulante
        }

        texto_postulante = generar_texto_postulante(datos_postulante)

        embedding_postulante = modelo.encode(texto_postulante)

        embedding_puesto = joblib.load(ARCHIVO_PUESTO_PKL)

        similitud = cosine_similarity(
            [embedding_puesto],
            [embedding_postulante]
        )[0][0]

        porcentaje = round(similitud * 100, 2)

        # Mostrar resultado

        st.subheader("📊 Resultado de la Evaluación")

        st.metric("Nivel de coincidencia", f"{porcentaje} %")

        if similitud >= 0.70:

            st.success("✅ POSTULANTE APTO")

        else:

            st.error("❌ POSTULANTE NO APTO")

        # Guardar resultado en CSV

        datos_postulante["similitud"] = porcentaje

        df_postulante = pd.DataFrame([datos_postulante])

        if os.path.exists(ARCHIVO_POSTULANTES):

            df_postulante.to_csv(
                ARCHIVO_POSTULANTES,
                mode="a",
                header=False,
                index=False
            )

        else:

            df_postulante.to_csv(
                ARCHIVO_POSTULANTES,
                index=False
            )

        st.info("📁 Resultado guardado correctamente")


# ---------------- INFO FINAL ----------------

st.info(
    "Este sistema utiliza Inteligencia Artificial y NLP para comparar "
    "el perfil del postulante con el perfil del puesto mediante similitud semántica."
)
