# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM TRES ARROYOS 2026
# Actualización: Lógica Híbrida (Corrección Hídrica) + Heatmap
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
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
    [data-testid="stSidebar"] {
        background-color: #dcfce7; 
        border-right: 1px solid #bbf7d0;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. LÓGICA TÉCNICA (ANN HÍBRIDA + BIO)
# ---------------------------------------------------------

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        """
        Xreal: [Julian, TMAX, TMIN, Prec]
        Implementa corrección agronómica para evitar inhibición por lluvia moderada.
        """
        Xn = self.normalize(Xreal)
        emer = []
        
        for i, x in enumerate(Xn):
            # A. Predicción Base (Neuronal)
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            val_ann = (np.tanh(z2) + 1) / 2
            
            # B. Lógica Híbrida (Parche Agronómico vK4.2)
            julian_real = Xreal[i, 0]
            tmin_real = Xreal[i, 2]
            prec_real = Xreal[i, 3]
            
            # Si llueve >= 5mm, es febrero (>35) y la noche es templada (>=14°C)
            # Aseguramos que el modelo no se 'asuste' por el agua.
            if prec_real >= 5.0 and julian_real > 35 and tmin_real >= 14.0:
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
def load_models():
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
# 3. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.title("⚙️ PREDWEEM vK4.2")
archivo_usuario = st.sidebar.file_uploader("Subir Clima (CSV/XLSX)", type=["xlsx", "csv"])

# Parámetros Sidebar
umbral_er = st.sidebar.slider("Umbral de Pico", 0.05, 0.90, 0.50)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima", value=20.0)
t_critica = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("Objetivo (°Cd)", value=600)

if archivo_usuario:
    # Carga de datos
    df = pd.read_csv(archivo_usuario) if archivo_usuario.name.endswith('.csv') else pd.read_excel(archivo_usuario)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'PREC': 'Prec', 'LLUVIA': 'Prec'}).dropna()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Ejecución Modelo Híbrido
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    # Bio-Cálculo
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    st.title("🌾 Monitor de Emergencia Corregido")
    
    # 1. Heatmap de Riesgo
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"],
        colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]], showscale=False
    ))
    fig_risk.update_layout(height=100, margin=dict(t=20, b=20), title="Mapa de Calor de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    # 2. Gráfico Principal
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Fecha"], y=df["EMERREL"], name="Emergencia Diaria", marker_color="#166534"))
    fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
    fig.update_layout(title="Dinámica de Emergencia (Lógica Híbrida vK4.2)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Ventana de Control
    indices_pico = df.index[df["EMERREL"] >= umbral_er].tolist()
    if indices_pico:
        f_inicio = df.loc[indices_pico[0], "Fecha"]
        df_v = df[df["Fecha"] >= f_inicio].copy()
        dga_actual = df_v["DG"].sum()
        
        st.success(f"✅ **Primer Pico Detectado:** {f_inicio.strftime('%d-%m-%Y')}")
        
        c1, c2 = st.columns(2)
        c1.metric("TT Acumulado", f"{dga_actual:.1f} °Cd")
        progreso = min(100.0, (dga_actual / dga_optimo) * 100)
        c2.progress(progreso/100)
        c2.write(f"Progreso hacia objetivo: {progreso:.1f}%")
    else:
        st.warning("⏳ Sin picos detectados con el umbral actual.")

    # Exportación
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Excel vK4.2", output.getvalue(), "Reporte_Hibrido.xlsx")

else:
    st.info("Cargue un archivo CSV para activar el modelo híbrido.")
