import streamlit as st
import joblib
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import traceback
from streamlit_lottie import st_lottie
import requests

# -------------------- CONFIG --------------------
# Se corrigió el caracter U+00A0 en esta sección
st.set_page_config(page_title="Sistema de Predicción de Riesgo de Diabetes",
                   layout="wide",
                   initial_sidebar_state="expanded")

UMBRAL_POR_DEFECTO = 0.35
REGISTRO_FILE = "registros_riesgo.csv"

# -------------------- ESTILOS --------------------
st.markdown("""
<style>
.main .block-container{padding-top:1rem;}
.card {
    padding: 18px;
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #d6eefc;
    box-shadow: 0 2px 6px rgba(3, 77, 128, 0.06);
    margin-bottom: 12px;
}
.small-muted { color: #456b7a; font-size: 13px; }
.stButton>button { background-color: #007acc !important; color: white !important; border-radius: 10px; height: 44px; font-size: 16px; border: none; }
.metric-text { font-size: 14px; }

[data-testid="stAppViewContainer"] {
    background-image: url("https://www.osdop.org.ar/wp-content/uploads/2022/08/Diabetes-1-1024x683.jpg"); 
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.0);
}

.block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 1.5rem 2rem;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- CARGAR MODELO Y SCALER --------------------
@st.cache_resource
def cargar_modelo_y_scaler(model_path='modelo_diabetes_dieta.pkl', scaler_path='scaler_datos_dieta.pkl'):
    """
    Devuelve (modelo, scaler, mensaje_error). Si hay error devuelve None,None y el mensaje.
    """
    try:
        if not os.path.exists(model_path):
            return None, None, f"No encontrado: {model_path}"
        if not os.path.exists(scaler_path):
            return None, None, f"No encontrado: {scaler_path}"
        modelo = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return modelo, scaler, None
    except Exception as e:
        return None, None, f"Error cargando modelos: {e}"

modelo, scaler, error_carga = cargar_modelo_y_scaler()
if error_carga:
    st.sidebar.error(error_carga)
    st.sidebar.info("Coloca 'modelo_diabetes_dieta.pkl' y 'scaler_datos_dieta.pkl' en la carpeta del proyecto.")
    st.stop()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙ Configuración")
    umbral = st.slider("Umbral decisión (sensibilidad vs especificidad)", 0.0, 1.0, UMBRAL_POR_DEFECTO, step=0.01)
    guardar_csv = st.checkbox("Guardar cada evaluación en registros (CSV)", value=True)
    descargar_historial = st.button("Descargar registros (CSV)")
    limpiar_registros = st.button("🗑 Limpiar registros")
    st.write("---")
    st.header("ℹ Sobre la app")
    st.write("Este sistema usa un modelo entrenado para estimar riesgo de diabetes basado en dieta y hábitos.")
    st.caption("Umbral configurable para experimentar trade-off entre sensibilidad y especificidad.")
    st.write("---")
    modo_avanzado = st.checkbox("Mostrar opciones avanzadas", value=False)

# Descargar registros si existe
if descargar_historial:
    if os.path.exists(REGISTRO_FILE):
        with open(REGISTRO_FILE, "rb") as f:
            st.download_button("Descargar CSV de registros", f, file_name=REGISTRO_FILE, mime="text/csv")
    else:
        st.info("Aún no hay registros guardados.")

# Limpiar registros (confirmación sencilla)
if limpiar_registros:
    if os.path.exists(REGISTRO_FILE):
        try:
            os.remove(REGISTRO_FILE)
            st.success("Registros eliminados.")
        except Exception as e:
            st.error(f"No se pudo eliminar: {e}")
    else:
        st.info("No existe el archivo de registros.")

# -------------------- UTILIDADES --------------------
SCORE_MAPPING = {
    'Sí (Diariamente)': (7, 1),
    'A Veces (1-3 veces/sem)': (4, 4),
    'No (Casi Nunca)': (1, 7)
}

# Mapeo de hábitos de vida a escala de riesgo (1-7)
HABITO_MAPPING = {
    'alcohol': {
        'No': 1,
        'Ocasionalmente': 4,
        'Regularmente (2+ veces/sem)': 7
    },
    'tabaco': {
        'No': 1,
        'Ex-fumador reciente': 4,
        'Regularmente (2+ veces/sem)': 7
    },
    'actividad': {
        'Regularmente (2+ veces/sem)': 1, # Menor riesgo = 1
        'Ocasionalmente': 4,
        'No': 7 # Mayor riesgo = 7
    }
}

def predecir_riesgo(datos_lista, model, scaler_obj, umbral_local):
    """
    datos_lista: lista con las 10 columnas en el orden esperado por el modelo.
    Retorna (pred, prob, detalle_error) donde prob está en [0,1].
    Maneja modelos que no tengan predict_proba.
    """
    try:
        # Se mantienen los nombres para el registro, pero el escalador usará .values
        columnas = [
            'glucosa', 'imc', 'edad', 'presion_arterial',
            'frec_comida_rapida', 'frec_azucar_añadido', 'frec_vegetales',
            # Se usan nombres genéricos para el registro
            'riesgo_alcohol_score', 'riesgo_tabaco_score', 'riesgo_actividad_score'
        ]
        
        # Se asegura que la lista de datos tenga la longitud correcta
        if len(datos_lista) != len(columnas):
             return None, None, f"Error: Se esperaban {len(columnas)} características pero se recibieron {len(datos_lista)}."

        # Crear DataFrame con nombres (solo para visualización y registro)
        df = pd.DataFrame([datos_lista], columns=columnas)
        
        # Validaciones básicas
        if scaler_obj is None:
            return None, None, "Scaler no está disponible."
            
        # CORRECCIÓN: Se usa .values para convertir el DataFrame en una matriz NumPy.
        # Esto ignora los nombres de las columnas, obligando al Scaler a usar el ORDEN de las 10 características.
        Xs = scaler_obj.transform(df.values) 
        
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(Xs)[:, 1][0])
        elif hasattr(model, "decision_function"):
            # convertir decision_function a probabilidad aproximada vía sigmoid
            score = float(model.decision_function(Xs)[0])
            prob = 1 / (1 + np.exp(-score))
        elif hasattr(model, "predict"):
            pred_raw = int(model.predict(Xs)[0])
            # si sólo tenemos predicción binaria, aproximamos prob a 0/1
            prob = float(pred_raw)
        else:
            return None, None, "El modelo no soporta predict/predict_proba/decision_function."
            
        pred = 1 if prob >= umbral_local else 0
        return pred, prob, None
    except Exception as e:
        tb = traceback.format_exc()
        # Muestra el error de dimensión si es el caso
        if "features" in str(e):
             # Si llega aquí, el error ya no debería ser por el nombre de la característica si se usa .values
             return None, None, f"Error de dimensión/estructura. Asegúrese de que el modelo y la predicción usan 10 características y el orden es correcto. {e}"
        return None, None, f"Error en predicción: {e}\n{tb}"

def guardar_registro_csv(data_dict):
    df = pd.DataFrame([data_dict])
    try:
        header = not os.path.exists(REGISTRO_FILE)
        df.to_csv(REGISTRO_FILE, mode='a', header=header, index=False)
    except Exception as e:
        return str(e)

def generar_pdf_bytes(data_dict):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 720)
    text.setFont("Helvetica-Bold", 14)
    text.textLine("Reporte de Predicción de Riesgo - Diabetes")
    c.drawText(text)
    y = 680
    c.setFont("Helvetica", 11)
    keys_to_exclude = ['riesgo_alcohol_score', 'riesgo_tabaco_score', 'riesgo_actividad_score', 'frec_comida_rapida_agg', 'frec_azucar_añadido_agg', 'frec_vegetales_agg']
    
    data_to_print = {}
    for k, v in data_dict.items():
        if k not in keys_to_exclude:
            display_name = k.replace('_', ' ').title().replace('Imc', 'IMC').replace('Pa', 'PA')
            data_to_print[display_name] = v

    for k, v in data_to_print.items():
        line = f"{k}: {v}"
        c.drawString(40, y, line)
        y -= 18
        if y < 80:
            c.showPage()
            y = 720
            c.setFont("Helvetica", 11)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
st.markdown("""
<h1 style='text-align:center; font-size:42px; color:#003b6f;'>
Sistema de Predicción de Riesgo de Diabetes
</h1>
""", unsafe_allow_html=True)
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_asistente = load_lottie("https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json")

col1, col2 = st.columns([5, 1])

with col1:
    st.markdown("""
    <div style='padding:20px; border-radius:12px; background:#eaf6ff; border-left:6px solid #007acc'>
        <h2>Bienvenidos</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_asistente, height=160)

st.write("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#fff3cd; padding:18px; border-radius:12px; border-left:6px solid #ffcc00; margin-top:10px;">
    <h3>¿Qué hace esta herramienta?</h3>
    <p style='font-size:16px'>
    Analiza tu alimentación, tus hábitos y tus signos vitales para darte una estimación
    personalizada del riesgo de diabetes y ofrecerte recomendaciones útiles.
    </p>
</div>
<br>
""", unsafe_allow_html=True)


# -------------------- LAYOUT --------------------
st.markdown(f"""
<h1 style=' color:#555; margin-top:-10px;'>
Sistema de Predicción de Riesgo de Diabetes
</h1>
""", unsafe_allow_html=True)
st.subheader(f"Umbral actual: {umbral:.2f} — Ajustable en la barra lateral")

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card"><h3>1. Datos personales y signos</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input('Edad (años)', min_value=18, max_value=120, value=30, step=1)
        peso = st.number_input('Peso (kg)', min_value=30.0, max_value=300.0, value=70.0, format="%.1f")
        altura = st.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
        sintoma_glucosa = st.slider('¿Sufre de sed excesiva o micción frecuente? (simula glucosa)', 0, 100, 30)
    with c2:
        diagnostico_hipertension = st.radio('¿Ha sido diagnosticado con Hipertensión?', ('No', 'Sí'))
        imc = round((peso / (altura ** 2)) if altura > 0 else 0.0, 2)
        st.metric(label="IMC Calculado (kg/m²)", value=f"{imc:.2f}")

    st.markdown('<div class="card"><h3>2. Evaluación Dietaria (Desayuno / Almuerzo / Cena)</h3></div>', unsafe_allow_html=True)
    opciones = list(SCORE_MAPPING.keys())
    with st.expander("Desayuno"):
        st.markdown("<p class='subtext'>Evalúa qué tan saludable suele ser tu desayuno.</p>", unsafe_allow_html=True)

        # D1
        st.markdown("*D1. Carbohidratos refinados?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en pan blanco, masitas, fideos, arroz blanco o galletas.</p>", unsafe_allow_html=True)
        d_riesgo = st.radio("", opciones, index=2, key="d_riesgo")

        # D2
        st.markdown("*D2. Añade azúcar a bebidas?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en café o té con azúcar, jugos endulzados o bebidas dulces.</p>", unsafe_allow_html=True)
        d_azucar = st.radio("", opciones, index=2, key="d_azucar")

        # D3
        st.markdown("*D3. Incluye proteína o fruta?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en huevo, yogurt, queso, pollo, manzana, banana, frutas frescas.</p>", unsafe_allow_html=True)
        d_proteccion = st.radio("", opciones, index=0, key="d_proteccion")

    # ---------------- ALMUERZO -----------------

    with st.expander("Almuerzo"):
        st.markdown("<p class='subtext'>Evalúa qué tan saludable suele ser tu almuerzo.</p>", unsafe_allow_html=True)

        # A1
        st.markdown("*A1. Consume frituras/procesados?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en pollo frito, papas fritas, embutidos, salchipapas, comida rápida.</p>", unsafe_allow_html=True)
        a_riesgo = st.radio("", opciones, index=2, key="a_riesgo")

        # A2
        st.markdown("*A2. Consume gaseosas/jugos procesados?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Incluye refrescos embotellados, jugos envasados, té helado dulce o gaseosas.</p>", unsafe_allow_html=True)
        a_azucar = st.radio("", opciones, index=2, key="a_azucar")

        # A3
        st.markdown("*A3. Incluye vegetales/legumbres?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en ensaladas, verduras cocidas, lentejas, arvejas o porotos.</p>", unsafe_allow_html=True)
        a_proteccion = st.radio("", opciones, index=0, key="a_proteccion")

    # ---------------- CENA -----------------

    with st.expander("Cena"):
        st.markdown("<p class='subtext'>Evalúa qué tan saludable suele ser tu cena.</p>", unsafe_allow_html=True)

        # C1
        st.markdown("*C1. Cena alta en grasas?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en carnes fritas, comida rápida, pizzas, empanadas o frituras.</p>", unsafe_allow_html=True)
        c_riesgo = st.radio("", opciones, index=2, key="c_riesgo")

        # C2
        st.markdown("*C2. Consume postres dulces?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Incluye helado, tortas, gelatina dulce, galletas o chocolates.</p>", unsafe_allow_html=True)
        c_azucar = st.radio("", opciones, index=2, key="c_azucar")

        # C3
        st.markdown("*C3. Cena rica en verduras/proteína?*", unsafe_allow_html=True)
        st.markdown("<p class='subtext'>Piensa en pollo a la plancha, verduras cocidas, ensaladas, pescado o huevo.</p>", unsafe_allow_html=True)
        c_proteccion = st.radio("", opciones, index=0, key="c_proteccion")


    st.markdown('<div class="card"><h3>3. Hábitos de estilo de vida</h3></div>', unsafe_allow_html=True)
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        consumo_alcohol = st.radio('¿Consume alcohol?', ('No', 'Ocasionalmente', 'Regularmente (2+ veces/sem)'), key="consumo_alcohol")
        actividad_fisica = st.radio('¿Realiza actividad física?', ('No', 'Ocasionalmente', 'Regularmente (2+ veces/sem)'), key="actividad_fisica")
    with c_h2:
        fumador = st.radio('¿Fumador / exfumador reciente?', ('No', 'Ex-fumador reciente', 'Regularmente (2+ veces/sem)'), key="fumador")

    st.write("")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("✅ Predecir Riesgo"):
            # --- Normalizaciones y mapeos ---
            glucosa_simulada = round(80 + float(sintoma_glucosa), 1)
            pa_simulada = 90 if diagnostico_hipertension == 'Sí' else 70

            # Mapeo de factores dietarios agregados (1-7)
            risk_inputs = [d_riesgo, a_riesgo, c_riesgo]
            total_risk_score = sum(SCORE_MAPPING[ch][0] for ch in risk_inputs)
            frec_comida_rapida = int(round(total_risk_score / 3))

            sugar_inputs = [d_azucar, a_azucar, c_azucar]
            total_sugar_score = sum(SCORE_MAPPING[ch][0] for ch in sugar_inputs)
            frec_azucar_añadido = int(round(total_sugar_score / 3))

            protection_inputs = [d_proteccion, a_proteccion, c_proteccion]
            total_protection_score = sum(SCORE_MAPPING[ch][1] for ch in protection_inputs)
            frec_vegetales = int(round(total_protection_score / 3))
            
            # Mapeo de Hábitos de vida a escala (1-7)
            riesgo_alcohol = HABITO_MAPPING['alcohol'][consumo_alcohol]
            riesgo_tabaco = HABITO_MAPPING['tabaco'][fumador]
            riesgo_actividad = HABITO_MAPPING['actividad'][actividad_fisica]
            # Codificación
            alcohol_map = {"No": 0, "Ocasionalmente": 1, "Regularmente (2+ veces/sem)": 2}
            actividad_map = {"No": 0, "Ocasionalmente": 1, "Regularmente (2+ veces/sem)": 2}
            fumador_map = {"No": 0, "Ex-fumador reciente": 1, "Regularmente (2+ veces/sem)": 2}

            datos_paciente = [
                float(glucosa_simulada),
                float(imc),
                int(edad),
                int(pa_simulada),
                frec_comida_rapida,
                frec_azucar_añadido,
                frec_vegetales,
                alcohol_map[consumo_alcohol],
                actividad_map[actividad_fisica],
                fumador_map[fumador]
            ]
            pred, prob, err = predecir_riesgo(datos_paciente, modelo, scaler, umbral)
            st.write("---")
            st.subheader("Resultados")
            if err:
                st.error(f"Error al predecir: {err}")
                st.stop()
            st.metric("Probabilidad estimada (P(Diabetes))", f"{prob*100:.2f} %")
            st.caption(f"Glucosa simulada: {glucosa_simulada:.0f} mg/dL | PA simulada: {pa_simulada} mmHg")
            if pred == 1:
                st.error("🔴 Clasificación: Alto Riesgo — Consulte a un profesional de salud.")
            else:
                st.success("🟢 Clasificación: Bajo Riesgo — Mantener hábitos saludables.")
            recomendaciones = []
            if imc >= 30:
                recomendaciones.append("IMC alto: considerar evaluación nutricional y plan de pérdida de peso.")
            elif imc >= 25:
                recomendaciones.append("IMC en rango de sobrepeso: actividad física regular y control dietético.")
            if actividad_fisica != 'Regularmente (2+ veces/sem)':
                recomendaciones.append("Iniciar actividad física: 30 min diarios, 5 días/semana.")
            if fumador != 'No':
                recomendaciones.append("Fumar incrementa riesgo; considere apoyo para dejar de fumar.")
            if frec_azucar_añadido >= 5:
                recomendaciones.append("Reducir consumo de azúcares añadidos.")
            if recomendaciones:
                st.markdown("Recomendaciones:")
                for r in recomendaciones:
                    st.write("- " + r)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidad (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#007acc"},
                    'steps': [
                        {'range': [0, umbral * 100], 'color': "#9be7b6"},
                        {'range': [umbral * 100, 70], 'color': "#ffd97d"},
                        {'range': [70, 100], 'color': "#ff9b9b"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            factores = pd.DataFrame({
                'Factor': ['Comida rápida (riesgo)', 'Azúcar añadido', 'Protección (vegetales)'],
                'Valor': [frec_comida_rapida, frec_azucar_añadido, frec_vegetales]
            })
            fig_bar = px.bar(factores, x='Factor', y='Valor', text='Valor', range_y=[0,7])
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- Guardar registro (INCLUYENDO CAMPOS DETALLADOS) ---
            timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
            registro = {
                'timestamp': timestamp,
                'edad': int(edad),
                'peso': float(peso),
                'altura': float(altura),
                'imc': round(imc, 2),
                'diagnostico_hipertension': diagnostico_hipertension,
                'sintoma_glucosa_slider': int(sintoma_glucosa),
                'glucosa_simulada': round(glucosa_simulada, 1),
                'pa_simulada': int(pa_simulada),
                
                # --- RESPUESTAS CRUDAS DE DIETA (9 campos) ---
                'd_riesgo': d_riesgo,
                'd_azucar': d_azucar,
                'd_proteccion': d_proteccion,
                'a_riesgo': a_riesgo,
                'a_azucar': a_azucar,
                'a_proteccion': a_proteccion,
                'c_riesgo': c_riesgo,
                'c_azucar': c_azucar,
                'c_proteccion': c_proteccion,
                
                # --- RESPUESTAS CRUDAS DE ESTILO DE VIDA (3 campos) ---
                'consumo_alcohol': consumo_alcohol,
                'actividad_fisica': actividad_fisica,
                'fumador': fumador,

                # --- VARIABLES AGREGADAS DE DIETA (usadas en el modelo) ---
                'frec_comida_rapida_agg': int(frec_comida_rapida),
                'frec_azucar_añadido_agg': int(frec_azucar_añadido),
                'frec_vegetales_agg': int(frec_vegetales),          
                # --- VARIABLES AGREGADAS DE HÁBITOS (usadas en el modelo) ---
                'riesgo_alcohol_score': riesgo_alcohol,
                'riesgo_tabaco_score': riesgo_tabaco,
                'riesgo_actividad_score': riesgo_actividad,
                'probabilidad': float(prob),
                'prediccion': int(pred),
                'umbral_usado': float(umbral)
            }
            if guardar_csv:
                error_guardado = guardar_registro_csv(registro)
                if error_guardado:
                    st.warning("No se pudo guardar el registro: " + error_guardado)
                else:
                    st.success("Registro guardado correctamente.")
            pdf_bytes = generar_pdf_bytes(registro)
            safe_ts = timestamp.replace(":", "-").replace(" ", "_")
            st.download_button("📄 Descargar reporte (PDF)", data=pdf_bytes,
                               file_name=f"reporte_{safe_ts}.pdf", mime="application/pdf")
    with col_b:
        st.info("Consejo: Ajusta el umbral en la barra lateral para priorizar sensibilidad (reducir falsos negativos) o especificidad.")
with right:
    st.markdown('<div class="card"><h3>Panel rápido</h3></div>', unsafe_allow_html=True)
    st.write("Acciones rápidas y últimos registros guardados.")
    if os.path.exists(REGISTRO_FILE):
        try:
            df_hist = pd.read_csv(REGISTRO_FILE)
            if not df_hist.empty:
                st.write("Últimos registros:")
                cols_to_display = ['timestamp', 'edad', 'imc', 'glucosa_simulada', 
                                   'frec_comida_rapida_agg', 'prediccion', 'probabilidad']
                existing_cols = [c for c in cols_to_display if c in df_hist.columns]
                
                st.dataframe(df_hist.tail(5).sort_values('timestamp', ascending=False)[existing_cols])
            else:
                st.write("Aún no hay registros guardados.")
        except Exception as e:
            st.error(f"Error leyendo registros: {e}")
    else:
        st.write("Aún no hay registros guardados.")
    if modo_avanzado:
        st.markdown("Opciones avanzadas")
        st.write("- Mostrar matriz de confusión y métricas (si dispone de conjunto de test).")
        st.write("- Exportar dataset para reentrenamiento.")
        try:
            st.markdown("Info modelo")
            st.write(f"Clase del modelo: {type(modelo)._name_}")
            st.write(f"Scaler: {type(scaler)._name_}")
        except Exception:
            st.info("No se pudo obtener el tipo de clase de modelo/scaler.")


st.write("---")
st.caption("Esta herramienta es informativa y NO reemplaza una evaluación clínica. Consulte a profesionales de salud para diagnóstico.")