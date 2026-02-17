# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM TRES ARROYOS 2026
# Actualización: Carga Automática + Lógica Híbrida de Lluvia
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import os
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4.2", 
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .bio-alert { padding: 10px; border-radius: 5px; background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; margin-bottom: 10px; font-size: 0.9em; }
</style>
""", unsafe_allow_True=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. MOTOR TÉCNICO (ANN HÍBRIDA + BIO)
# ---------------------------------------------------------

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for i, x in enumerate(Xn):
            # A. Predicción Neural
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            val_ann = (np.tanh(z2) + 1) / 2
            
            # B. Lógica Híbrida (Corrección Agronómica vK4.2)
            julian, tmin, prec = Xreal[i, 0], Xreal[i, 2], Xreal[i, 3]
            # Si llueve >= 5mm en febrero con noches templadas, forzamos emergencia
            if prec >= 5.0 and julian > 35 and tmin >= 14.0:
                val_final = max(float(val_ann), 0.85)
            else:
                val_final = float(val_ann)
            emer.append(val_final)
            
        emer = np.array(emer).flatten()
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else: return 0.0

@st.cache_resource
def load_assets():
    try:
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except:
        return None, None

# ---------------------------------------------------------
# 3. GESTIÓN DE DATOS (CARGA AUTOMÁTICA)
# ---------------------------------------------------------
def load_data(uploaded_file):
    # Intentar cargar archivo subido o el archivo local automático
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.sidebar.success("✅ Archivo subido cargado.")
    elif os.path.exists(BASE / "meteo_daily.csv"):
        df = pd.read_csv(BASE / "meteo_daily.csv")
        st.sidebar.info("📂 Cargado automáticamente: 'meteo_daily.csv'")
    else:
        return None

    # Normalizar columnas
    df.columns = [c.upper().strip() for c in df.columns]
    mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
    df = df.rename(columns=mapeo).dropna(subset=['TMAX', 'TMIN'])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 4. INTERFAZ Y EJECUCIÓN
# ---------------------------------------------------------
modelo_ann, cluster_model = load_assets()

# Sidebar
st.sidebar.title("🌾 PREDWEEM vK4.2")
archivo_subido = st.sidebar.file_uploader("Actualizar Clima (opcional)", type=["xlsx", "csv"])
umbral_er = st.sidebar.slider("Umbral de Pico", 0.10, 0.90, 0.50)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima", value=20.0)
t_critica = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("Objetivo Control (°Cd)", value=600)

# Cargar Datos
df = load_data(archivo_subido)

if df is not None and modelo_ann is not None:
    # Procesamiento
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    # Cálculos Bio-Térmicos
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # --- UI PRINCIPAL ---
    st.title("🌾 Dashboard de Emergencia Híbrido")
    
    # Mapa de Calor
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"],
        colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]], showscale=False
    ))
    fig_risk.update_layout(height=100, margin=dict(t=20, b=20), title="Intensidad de Nacimientos")
    st.plotly_chart(fig_risk, use_container_width=True)

    # Gráfico de Barras
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Fecha"], y=df["EMERREL"], name="Tasa Diaria", marker_color="#166534"))
    fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text="Umbral de Pico")
    fig.update_layout(title="Dinámica de Emergencia (vK4.2)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Ventana de Seguimiento
    indices_pico = df.index[df["EMERREL"] >= umbral_er].tolist()
    if indices_pico:
        f_inicio = df.loc[indices_pico[0], "Fecha"]
        dga_actual = df[df["Fecha"] >= f_inicio]["DG"].sum()
        
        col1, col2 = st.columns(2)
        col1.metric("Inicio de Ventana", f_inicio.strftime('%d-%m-%Y'))
        col2.metric("Acumulado Post-Pico", f"{dga_actual:.1f} °Cd")
        
        progreso = min(100.0, (dga_actual / dga_optimo) * 100)
        st.write(f"**Progreso hacia aplicación óptima ({dga_optimo} °Cd):**")
        st.progress(progreso/100)
    else:
        st.warning("⚠️ No se detectan picos de emergencia significativos con los datos actuales.")

    # Descarga
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Reporte Completo", output.getvalue(), "PREDWEEM_Final_Report.xlsx")

else:
    st.warning("❌ No se encontró 'meteo_daily.csv' ni se subió un archivo manual.")
