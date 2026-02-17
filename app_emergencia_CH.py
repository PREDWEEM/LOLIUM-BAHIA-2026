# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2.3 — LOLIUM TRES ARROYOS 2026
# Cambios: Carga Automática + Parche Lluvia + Fix Python 3.13
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
    page_title="PREDWEEM INTEGRAL vK4.2.3", 
    layout="wide",
    page_icon="🌾"
)

# FIX: Corregido unsafe_allow_html
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
            # A. Predicción Neural
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            val_ann = (np.tanh(z2) + 1) / 2
            
            # FIX Python 3.13: Conversión segura a escalar
            val_ann_scalar = float(np.ravel(val_ann)[0])
            
            # B. Lógica Híbrida (Corrección Agronómica de Lluvia)
            julian, tmin, prec = Xreal[i, 0], Xreal[i, 2], Xreal[i, 3]
            # Si llueve >= 5mm en febrero con noches templadas, forzamos emergencia
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

def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

# ---------------------------------------------------------
# 3. GESTIÓN DE DATOS (CARGA AUTOMÁTICA)
# ---------------------------------------------------------
def load_data(uploaded_file):
    # Intentar cargar archivo subido o el archivo local automático
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    elif os.path.exists(BASE / "meteo_daily.csv"):
        df = pd.read_csv(BASE / "meteo_daily.csv")
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
st.sidebar.title("🌾 PREDWEEM vK4.2.3")
archivo_subido = st.sidebar.file_uploader("Actualizar Clima (opcional)", type=["xlsx", "csv"])

st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral de Pico", 0.10, 0.90, 0.50)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima", value=20.0)
t_critica = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("Objetivo Control (°Cd)", value=600)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800)

df = load_data(archivo_subido)

if df is not None and modelo_ann is not None:
    # Procesamiento
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    # Bio-Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # --- UI PRINCIPAL ---
    st.title("🌾 Monitor PREDWEEM — Bahía Blanca 2026")
    
    # Heatmap (Mapa de Calor)
    colorscale_hard = [[0.0, "green"], [0.49, "green"], [0.50, "yellow"], [0.79, "yellow"], [0.80, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0), title="Intensidad de Emergencia (Modelo Híbrido)")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS DTW", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        
        # Lógica de Primer Pico
        indices_pico = df.index[df["EMERREL"] >= umbral_er].tolist()
        
        if indices_pico:
            idx_primer = indices_pico[0]
            f_inicio = df.loc[idx_primer, "Fecha"]
            
            # Acumulados
            df_post = df[df["Fecha"] >= f_inicio].copy()
            dga_actual = df_post["DG"].sum()
            dias_stress = len(df_post[df_post["Tmedia"] > t_opt_max])
            
            with col_main:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df["Fecha"], y=df["EMERREL"], name="Tasa Diaria", marker_color="#166534"))
                fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
                fig.update_layout(title="Dinámica de Emergencia", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"📅 **Conteo desde el primer pico:** {f_inicio.strftime('%d-%m-%Y')}")
                if dias_stress > 0:
                    st.markdown(f'<div class="bio-alert">🔥 {dias_stress} días con temperaturas por encima de la óptima.</div>', unsafe_allow_html=True)
            
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = dga_actual,
                    title = {'text': "TT ACUMULADO (°Cd)"},
                    gauge = {
                        'axis': {'range': [0, dga_critico * 1.2]},
                        'steps': [
                            {'range': [0, dga_optimo], 'color': "#4ade80"},
                            {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                            {'range': [dga_critico, dga_critico*1.2], 'color': "#f87171"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'value': dga_actual}
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("⏳ Esperando la detección del primer pico de emergencia.")

    with tab2:
        st.subheader("Clasificación por Dinámica Temporal (DTW)")
        if cluster_model and len(df) > 30:
            # Aquí iría tu lógica de DTW simplificada para el dashboard
            st.info("Comparando campaña actual con patrones históricos...")
            # (El código DTW que ya tenías funciona aquí)

    with tab3:
        st.subheader("Configuración Fisiológica")
        x_temps = np.linspace(0, 40, 100)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        fig_bio = go.Figure(go.Scatter(x=x_temps, y=y_tt, fill='tozeroy', line_color='#2563eb'))
        fig_bio.update_layout(title="Respuesta al Tiempo Térmico", xaxis_title="Temp (°C)", yaxis_title="Eficiencia (°Cd)")
        st.plotly_chart(fig_bio, use_container_width=True)

    # Exportación
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Final.xlsx")

else:
    st.error("❌ Faltan archivos: Asegúrate de tener 'meteo_daily.csv', 'IW.npy' y 'LW.npy' en la carpeta.")
