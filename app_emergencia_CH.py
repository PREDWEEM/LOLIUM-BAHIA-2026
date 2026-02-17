# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2.4 — LOLIUM TRES ARROYOS 2026
# Corrección: Unpacking Error en load_assets + Fix Python 3.13
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
    page_title="PREDWEEM INTEGRAL vK4.2.4", 
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
""", unsafe_allow_html=True)

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
        for i in range(len(Xn)):
            x = Xn[i]
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            val_ann = (np.tanh(z2) + 1) / 2
            
            # --- FIX COMPATIBILIDAD PYTHON 3.13 ---
            val_ann_scalar = float(np.ravel(val_ann)[0])
            
            # Lógica Híbrida (Corrección Lluvia)
            julian, tmin, prec = Xreal[i, 0], Xreal[i, 2], Xreal[i, 3]
            if prec >= 5.0 and julian > 35 and tmin >= 14.0:
                val_final = max(val_ann_scalar, 0.85)
            else:
                val_final = val_ann_scalar
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
    """
    Carga los pesos de la red y el modelo de clusters.
    IMPORTANTE: Siempre devuelve una tupla de 2 elementos para evitar TypeError.
    """
    try:
        # Cargar ANN
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        # Cargar Clusters
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        # Si falla, devolvemos Nones pero en formato tupla (evita el error de la línea 132)
        return None, None

# ---------------------------------------------------------
# 3. GESTIÓN DE DATOS (CARGA AUTOMÁTICA)
# ---------------------------------------------------------
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    elif os.path.exists(BASE / "meteo_daily.csv"):
        df = pd.read_csv(BASE / "meteo_daily.csv")
    else:
        return None

    df.columns = [c.upper().strip() for c in df.columns]
    mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
    df = df.rename(columns=mapeo).dropna(subset=['TMAX', 'TMIN'])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 4. INTERFAZ Y EJECUCIÓN
# ---------------------------------------------------------

# Línea 132 (Aproximada): Ejecución segura del cargador
modelo_ann, cluster_model = load_assets()

st.sidebar.title("⚙️ Configuración")
archivo_subido = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
umbral_er = st.sidebar.slider("Umbral de Pico", 0.10, 0.90, 0.50)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima", value=20.0)
t_critica = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800)

df = load_data(archivo_subido)

if df is not None and modelo_ann is not None:
    # A. Predicción Neural
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
    emerrel, _ = modelo_ann.predict(X)
    
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    # B. Cálculo Bio-Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    st.title("🌾 Monitor PREDWEEM — Bahía Blanca 2026")
    
    # Visualización Heatmap
    colorscale_hard = [[0.0, "green"], [0.49, "green"], [0.50, "yellow"], [0.79, "yellow"], [0.80, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0), title="Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 MONITOR", "📈 ESTRATEGIA", "🧪 BIO-CONFIG"])

    with tab1:
        # Lógica de Inicio en el Primer Pico
        indices_pico = df.index[df["EMERREL"] >= umbral_er].tolist()
        
        if indices_pico:
            f_inicio = df.loc[indices_pico[0], "Fecha"]
            df_ventana = df[df["Fecha"] >= f_inicio].copy()
            dga_actual = df_ventana["DG"].sum()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', name="Emergencia"))
                fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Inicio de Ventana", f_inicio.strftime('%d-%m-%Y'))
                st.metric("TT Acumulado", f"{dga_actual:.1f} °Cd")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = dga_actual,
                    gauge = {'axis': {'range': [0, dga_critico*1.1]},
                             'steps': [{'range': [0, dga_optimo], 'color': "lightgreen"},
                                       {'range': [dga_optimo, dga_critico], 'color': "yellow"}]}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("Esperando el primer pico de emergencia...")

    # Exportación
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "Reporte_Lolium.xlsx")

elif modelo_ann is None:
    st.error("❌ No se pudieron cargar los archivos del modelo (.npy / .pkl). Verifica que estén en la carpeta raíz del repositorio.")
else:
    st.info("👋 Por favor, sube un archivo de clima para comenzar.")
