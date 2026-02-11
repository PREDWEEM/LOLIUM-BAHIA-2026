# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM BAHÍA BLANCA 2026 (Mobile-First)
# Mejoras: Layout centered + Tarjetas modernas + CSS responsive full
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
    page_title="PREDWEEM INTEGRAL vK4.2",
    layout="centered",  # Clave para móvil
    page_icon="🌾",
    initial_sidebar_state="auto"
)

# CSS moderno y 100% responsive
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
        padding: 1rem 0;
    }
    [data-testid="stSidebar"] {
        background-color: #dcfce7;
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    .bio-alert {
        padding: 12px;
        border-radius: 12px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin: 10px 0;
        font-size: 1em;
    }
    #MainMenu, header, footer {visibility: hidden;}
    
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.4rem !important; }
        .stButton>button {
            width: 100% !important;
            height: 3.5rem !important;
            font-size: 1.1rem !important;
        }
        .block-container { padding: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Función card reutilizable
def card(title: str, icon: str = "📊"):
    st.markdown(f"<div class='card'><h3 style='color:#166534; margin-top:0;'>{icon} {title}</h3>", unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK
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
# 3. LÓGICA TÉCNICA
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else:
        return 0.0

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
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer).flatten() + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

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
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def get_data(file_input):
    try:
        if file_input:
            if file_input.name.endswith('.csv'):
                df = pd.read_csv(file_input, parse_dates=["Fecha"])
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path = BASE / "meteo_daily.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Fecha"])
            else:
                return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            'FECHA': 'Fecha', 'DATE': 'Fecha', 
            'TMAX': 'TMAX', 'TMIN': 'TMIN', 
            'PREC': 'Prec', 'LLUVIA': 'Prec'
        }
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

# ---------------------------------------------------------
# 4. CARGA MODELOS + SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Emergencia**")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Para detectar pico)", 0.05, 0.80, 0.50)

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

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0 
    
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    st.markdown("<h1 style='text-align:center; color:#166534;'>🌾 PREDWEEM LOLIUM 2026</h1>", unsafe_allow_html=True)
    
    # Heatmap
    colorscale_hard = [[0.0, "#86efac"], [0.49, "#86efac"], [0.50, "#fbbf24"], [0.79, "#fbbf24"], [0.80, "#f87171"], [1.0, "#f87171"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=150, margin=dict(t=40, b=20, l=10, r=10), title="Mapa de Intensidad de Emergencia", title_x=0.5)
    
    card("📈 Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
    card_end()
    
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        # Lógica primer pico
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        fecha_inicio_ventana = None
        if indices_pulso:
            first_peak_index = indices_pulso[0]
            fecha_inicio_ventana = df.loc[first_peak_index, "Fecha"]
        
        dga_actual = 0.0
        dias_stress = 0
        if fecha_inicio_ventana:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            dga_actual = df_ventana["DGA_cum"].iloc[-1] if not df_ventana.empty else 0.0
            dias_stress = len(df_ventana[df_ventana["Tmedia"] > t_opt_max])
        
        col_main, col_gauge = st.columns([2, 1])
        
        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria',
                line=dict(color='#166534', width=3), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.15)'
            ))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral ({umbral_er})")
            fig_emer.update_layout(title="Dinámica de Emergencia y Picos", height=400)
            
            card("📅 Dinámica de Emergencia")
            st.plotly_chart(fig_emer, use_container_width=True)
            card_end()
            
            if fecha_inicio_ventana:
                st.success(f"**Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico)")
                if dias_stress > 0:
                    st.markdown(f"<div class='bio-alert'>🔥 Estrés Térmico: {dias_stress} días con T > {t_opt_max}°C</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⏳ Esperando primer pico (≥ {umbral_er})")
        
        with col_gauge:
            fecha_hoy = pd.Timestamp("2026-02-11")  # Fecha actual fija para demo (cámbiala si querés dinámica)
            if fecha_hoy not in df['Fecha'].values:
                fecha_hoy = df['Fecha'].max()

            idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
            df_periodo_total = df.iloc[:idx_hoy + 8].copy()
            
            indices_pico = df_periodo_total.index[df_periodo_total["EMERREL"] >= umbral_er].tolist()
            
            dga_hoy = 0.0
            dga_7dias = 0.0
            msg_estado = "Esperando pico..."

            if indices_pico:
                idx_primer_pico = indices_pico[0]
                fecha_inicio_pico = df.loc[idx_primer_pico, "Fecha"]
                
                if fecha_inicio_pico <= fecha_hoy:
                    df_hasta_hoy = df[(df["Fecha"] >= fecha_inicio_pico) & (df["Fecha"] <= fecha_hoy)]
                    dga_hoy = df_hasta_hoy["DG"].sum()
                    df_pronostico = df.iloc[idx_hoy + 1 : idx_hoy + 8]
                    dga_7dias = dga_hoy + df_pronostico["DG"].sum()
                    msg_estado = f"Pico el {fecha_inicio_pico.strftime('%d/%m')}"
                else:
                    dga_hoy = 0.0
                    df_futuro_post_pico = df[(df["Fecha"] >= fecha_inicio_pico) & (df.index <= idx_hoy + 7)]
                    dga_7dias = df_futuro_post_pico["DG"].sum()
                    msg_estado = f"⚠️ Pico previsto {fecha_inicio_pico.strftime('%d/%m')}"
            
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", 
                value = dga_hoy,
                title = {'text': "<b>TT ACUMULADO (°Cd)</b>"},
                gauge = {
                    'axis': {'range': [None, max_axis]},
                    'bar': {'color': "#1e293b"},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#4ade80"},
                        {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                        {'range': [dga_critico, max_axis], 'color': "#f87171"}
                    ],
                    'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}
                }
            ))
            fig_gauge.add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>+7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14))
            fig_gauge.update_layout(height=400, margin=dict(t=80, b=50))
            
            card("🌡️ Gauge Térmico")
            st.plotly_chart(fig_gauge, use_container_width=True)
            card_end()

    with tab2:
        card("🔍 Clasificación DTW")
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()

        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)

            dists = []
            for m in cluster_model["curves_interp"]:
                m_slice = m[JD_COM <= jd_corte]
                m_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(obs_norm, m_norm))

            pred = int(np.argmin(dists))
            names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, use_container_width=True)
            with c2:
                st.success(f"### {names.get(pred)}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
            st.info("Datos insuficientes para clasificación DTW.")
        card_end()

    with tab3:
        card("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]
        
        fig_bio = go.Figure()
        fig_bio.add_trace(go.Scatter(x=x_temps, y=y_tt, mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'))
        fig_bio.add_vrect(x0=t_base_val, x1=t_opt_max, fillcolor="green", opacity=0.1, annotation_text="Óptimo")
        fig_bio.add_vrect(x0=t_opt_max, x1=t_critica, fillcolor="orange", opacity=0.1, annotation_text="Estrés")
        fig_bio.add_vrect(x0=t_critica, x1=45, fillcolor="red", opacity=0.1, annotation_text="Inhibición")
        fig_bio.update_layout(xaxis_title="T Media (°C)", yaxis_title="TT Diario (°Cd)", height=400)
        st.plotly_chart(fig_bio, use_container_width=True)
        
        st.info(f"**Interpretación:** Hasta {t_base_val}°C → 0. Entre {t_base_val}-{t_opt_max}°C → lineal. {t_opt_max}-{t_critica}°C → penalizado. >{t_critica}°C → stop.")
        card_end()

    # Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'Umbral_Pico'],
                      'Valor': [t_base_val, t_opt_max, t_critica, umbral_er]}).to_excel(writer, sheet_name='Bio_Params', index=False)
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Descargar Reporte Completo",
        data=output.getvalue(),
        file_name="PREDWEEM_Report_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👋 **Bienvenido a PREDWEEM.** Subí datos climáticos para comenzar.")
