# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM TRES ARROYOS 2026 (Mobile-First)
# Mejoras: Layout centered + Cards modernas + CSS responsive
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO MODERNO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4",
    layout="centered",  # ¡Clave para móvil!
    page_icon="🌾",
    initial_sidebar_state="auto"
)

# CSS moderno y responsive
st.markdown("""
<style>
    /* Fondo general */
    .main {
        background-color: #f8fafc;
        padding: 1rem 0;
    }
    
    /* Sidebar verde suave */
    [data-testid="stSidebar"] {
        background-color: #dcfce7;
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    
    /* Tarjetas modernas */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Alert personalizada */
    .bio-alert {
        padding: 12px;
        border-radius: 12px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin: 10px 0;
        font-size: 1em;
    }
    
    /* Ocultar elementos nativos */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Responsivo para móvil */
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.4rem !important; }
        
        .stButton>button {
            width: 100% !important;
            height: 3.5rem !important;
            font-size: 1.1rem !important;
        }
        
        .block-container {
            padding: 1rem !important;
        }
        
        /* Gráficos más altos en móvil */
        .plotly-graph-div { height: 400px !important; }
    }
</style>
""", unsafe_allow_html=True)

# Función card reutilizable
def card(title: str, content, icon: str = "📊"):
    st.markdown(f"""
    <div class="card">
        <h3 style="color:#166534; margin-top:0;">{icon} {title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK (sin cambios)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=150)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=150) - (np.arange(150)*0.1),
            "TMIN": np.random.uniform(10, 18, size=150) - (np.arange(150)*0.06),
            "Prec": np.random.choice([0, 0, 5, 15, 45], size=150)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (sin cambios importantes)
# ---------------------------------------------------------
# ... (mantengo todo igual: dtw_distance, calculate_tt_scalar, PracticalANNModel, load_models, get_data)

# (Copia aquí las funciones que ya tenías, sin cambios)

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Emergencia**")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Para detectar pico)", 0.05, 0.80, 0.50, help="Valor más alto = pico más exigente")

st.sidebar.divider()
st.sidebar.markdown("🌡️ **Fisiología Térmica (Bio-Limit)**")
st.sidebar.caption("Ajusta la respuesta biológica al calor.")

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=50)

# Botón de descarga más visible
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Reporte")
# (el download_button va al final)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    
    # (Todo el preprocesamiento igual)
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    # Título principal
    st.markdown("<h1 style='text-align:center; color:#166534;'>🌾 PREDWEEM LOLIUM 2026</h1>", unsafe_allow_html=True)
    
    # Heatmap en tarjeta
    colorscale_hard = [[0.0, "#86efac"], [0.49, "#86efac"], [0.50, "#fbbf24"], [0.79, "#fbbf24"], [0.80, "#f87171"], [1.0, "#f87171"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(
        height=150, margin=dict(t=40, b=20, l=10, r=10),
        title="Mapa de Intensidad de Emergencia",
        title_x=0.5
    )
    card("📈 Mapa de Intensidad de Emergencia", st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False}))
    
    # Tabs envueltos en cards
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        # (Lógica de ventana y gauge igual)
        # ... (copia tu lógica de primer pico, dga_hoy, etc.)

        col_main, col_gauge = st.columns([2, 1])
        
        with col_main:
            # Gráfico de emergencia en tarjeta
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria',
                line=dict(color='#166534', width=3), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.15)'
            ))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
            fig_emer.update_layout(title="Dinámica de Emergencia", height=400)
            card("📅 Dinámica de Emergencia y Picos", st.plotly_chart(fig_emer, use_container_width=True))
            
            # Info de ventana
            if fecha_inicio_ventana:
                st.success(f"**Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')}")
                if dias_stress > 0:
                    st.markdown(f"""<div class="bio-alert">🔥 Estrés Térmico: {dias_stress} días con T > {t_opt_max}°C</div>""", unsafe_allow_html=True)
            else:
                st.warning(f"Esperando primer pico (≥ {umbral_er})")

        with col_gauge:
            # Gauge en tarjeta (más grande en móvil por CSS)
            card("🌡️ TT Acumulado (°Cd)", st.plotly_chart(fig_gauge, use_container_width=True))

    # Tab2 y Tab3 también envueltos en cards donde tenga sentido
    # (Puedes hacer lo mismo: card("Clasificación DTW", contenido))

    # Exportación (botón más grande en sidebar)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'Umbral_Pico'],
                      'Valor': [t_base_val, t_opt_max, t_critica, umbral_er]}).to_excel(writer, sheet_name='Bio_Params', index=False)
    
    st.sidebar.download_button(
        label="📥 Descargar Reporte Completo",
        data=output.getvalue(),
        file_name="PREDWEEM_Report_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👋 **Bienvenido a PREDWEEM.** Subí tus datos climáticos para comenzar el análisis.")
