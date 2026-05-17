
# app.py — VitalGuard AI 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CONFIGURACIÓN DE PÁGINA

st.set_page_config(
    page_title="VitalGuard AI",
    page_icon="❤️",
    layout="wide"
)

# SIDEBAR

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
        width=120
    )

    st.title("VitalGuard AI")

    st.markdown("---")

    st.subheader("Información IA")

    st.write("Modelo: Random Forest")
    st.write("Tipo: Aprendizaje Supervisado")
    st.write("Clasificación Binaria")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("Heart Failure Clinical Records")
    st.write("299 registros clínicos")

    st.markdown("---")

    st.subheader("Equipo")

    st.write("• Sebastián")
    st.write("• Integrante 2")
    st.write("• Integrante 3")

    st.markdown("---")

    st.success("ODS 3 — Salud y Bienestar")

# ESTILOS PERSONALIZADOS

st.markdown(
    """
  st.markdown(
        font-size: 32px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .risk-low {
        background: linear-gradient(
            135deg,
            #14532d,
            #166534
        );
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }

    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="metric-container"] div {
        color: #f8fafc !important;
    }

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        color: white;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(37,99,235,0.35);
    }

    .stSelectbox label,
    .stSlider label,
    .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    hr {
        border-color: #1f2937;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">VitalGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Sistema Inteligente para Evaluación de Riesgo Cardiovascular</div>',
    unsafe_allow_html=True
)

# INFORMACIÓN IA

st.info(
    "Modelo IA: Random Forest Classifier | "
    "Tipo: Aprendizaje Supervisado | "
    "Problema: Clasificación Binaria"
)

# ENTRENAMIENTO DEL MODELO

@st.cache_resource
def train_model():

    # CARGAR DATASET
    df = pd.read_csv("data/heart.csv")

    # VARIABLES
    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    # DIVISIÓN
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODELO IA
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    # ENTRENAMIENTO
    model.fit(X_train, y_train)

    # EVALUACIÓN
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # IMPORTANCIA VARIABLES
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": importance
    })

    return model, accuracy, importance_df


model, accuracy, importance_df = train_model()

# MÉTRICAS SUPERIORES

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

with m2:
    st.metric("Variables", "12")

with m3:
    st.metric("Tipo IA", "Supervisado")


st.markdown("### Casos de Prueba")

demo1, demo2 = st.columns(2)

healthy_demo = False
critical_demo = False

with demo1:
    healthy_demo = st.button("Paciente Saludable")

with demo2:
    critical_demo = st.button("Paciente Crítico")

# DATOS AUTOMÁTICOS

if healthy_demo:

    default_age = 30
    default_anaemia = 0
    default_diabetes = 0
    default_pressure = 0
    default_smoking = 0
    default_sex = 1
    default_cpk = 120
    default_ejection = 60
    default_platelets = 260000.0
    default_creatinine = 0.9
    default_sodium = 140
    default_time = 200

elif critical_demo:

    default_age = 82
    default_anaemia = 1
    default_diabetes = 1
    default_pressure = 1
    default_smoking = 1
    default_sex = 1
    default_cpk = 900
    default_ejection = 20
    default_platelets = 120000.0
    default_creatinine = 4.0
    default_sodium = 120
    default_time = 5

else:

    default_age = 60
    default_anaemia = 0
    default_diabetes = 0
    default_pressure = 0
    default_smoking = 0
    default_sex = 1
    default_cpk = 250
    default_ejection = 35
    default_platelets = 250000.0
    default_creatinine = 1.2
    default_sodium = 137
    default_time = 120



col1, col2 = st.columns([1,1])

# FORMULARIO

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Información del Paciente")

    age = st.slider(
        "Edad",
        20,
        100,
        default_age,
        help="Edad del paciente."
    )

    anaemia = st.selectbox(
        "Anemia",
        [0, 1],
        index=default_anaemia,
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        index=default_diabetes,
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    high_blood_pressure = st.selectbox(
        "Presión Alta",
        [0, 1],
        index=default_pressure,
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    smoking = st.selectbox(
        "Fumador",
        [0, 1],
        index=default_smoking,
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    sex = st.selectbox(
        "Sexo",
        [0, 1],
        index=default_sex,
        format_func=lambda x: "Masculino" if x == 1 else "Femenino"
    )

    creatinine_phosphokinase = st.number_input(
        "Creatinine Phosphokinase",
        value=default_cpk
    )

    ejection_fraction = st.slider(
        "Fracción de Eyección",
        10,
        80,
        default_ejection
    )

    platelets = st.number_input(
        "Plaquetas",
        value=default_platelets
    )

    serum_creatinine = st.number_input(
        "Creatinina en Suero",
        value=default_creatinine
    )

    serum_sodium = st.slider(
        "Sodio en Suero",
        100,
        150,
        default_sodium
    )

    time = st.slider(
        "Tiempo de Seguimiento",
        1,
        300,
        default_time
    )

    analyze = st.button("Analizar Riesgo")

    st.markdown('</div>', unsafe_allow_html=True)


with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Resultado del Análisis")

    st.metric(
        "Accuracy del Modelo",
        f"{accuracy * 100:.2f}%"
    )

    if analyze:

        with st.spinner("Analizando paciente..."):

            patient_data = np.array([[
                age,
                anaemia,
                creatinine_phosphokinase,
                diabetes,
                ejection_fraction,
                high_blood_pressure,
                platelets,
                serum_creatinine,
                serum_sodium,
                sex,
                smoking,
                time
            ]])

            # PREDICCIÓN
            prediction = model.predict(patient_data)

            # PROBABILIDAD
            probability = model.predict_proba(patient_data)

            low_risk = probability[0][0] * 100
            high_risk = probability[0][1] * 100

            # RESULTADO
            if prediction[0] == 1:

                st.markdown(
                    f'''
                    <div class="risk-high">
                        ALTO RIESGO
                        <br>
                        {high_risk:.2f}%
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                st.error(
                    "El paciente presenta alta probabilidad de evento cardíaco."
                )

            else:

                st.markdown(
                    f'''
                    <div class="risk-low">
                         BAJO RIESGO
                        <br>
                        {low_risk:.2f}%
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                st.success(
                    "El paciente presenta bajo riesgo cardiometabólico."
                )

           
            # PROBABILIDADES
          

            st.markdown("---")

            st.subheader("Probabilidades")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Bajo Riesgo",
                    f"{low_risk:.2f}%"
                )

            with metric2:
                st.metric(
                    "Alto Riesgo",
                    f"{high_risk:.2f}%"
                )

            
            # GAUGE CHART

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_risk,
                title={'text': "Riesgo Cardíaco"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "#14532d"},
                        {'range': [40, 70], 'color': "#ca8a04"},
                        {'range': [70, 100], 'color': "#7f1d1d"}
                    ]
                }
            ))

            fig.update_layout(height=300)

            st.plotly_chart(
                fig,
                use_container_width=True
            )



            st.markdown("---")

            st.subheader("🧠 Variables Analizadas")

            st.write([
                "Edad",
                "Diabetes",
                "Presión Alta",
                "Creatinina",
                "Fracción de Eyección",
                "Plaquetas",
                "Sodio",
                "Tabaquismo"
            ])


            st.markdown("---")

            st.subheader("Variables Más Importantes")

            st.bar_chart(
                importance_df.set_index("Variable")
            )


            st.markdown("---")

            st.subheader("🩺 Recomendaciones Inteligentes")

            if prediction[0] == 1:

                st.warning("✔ Realizar chequeo médico prioritario")
                st.warning("✔ Monitorear presión arterial")
                st.warning("✔ Reducir consumo de sodio")
                st.warning("✔ Mantener seguimiento cardiológico")

            else:

                st.info("✔ Mantener hábitos saludables")
                st.info("✔ Continuar actividad física")
                st.info("✔ Realizar controles periódicos")

    else:

        st.info(
            "Ingrese los datos del paciente y presione 'Analizar Riesgo'."
        )

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")

st.caption(
    "VitalGuard AI © 2026 | Inteligencia Artificial aplicada a la Salud Preventiva"
)