import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

#Configuracion de la página
st.set_page_config(
    page_title="House Prices Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
#Carga de datos
@st.cache_data
def load_data():
    return pd.read_csv('data/train.csv')


df = load_data()

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #212529; }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #212529;
        margin: 1.5rem 0 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #f0f0f0;
    }
    [data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — filtros ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Filtros")
    barrio = st.multiselect("Barrio", df["Neighborhood"].unique())

# ── Filtrado ──────────────────────────────────────────────────────────────────
df_filtrado = df[df["Neighborhood"].isin(barrio)] if barrio else df

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏠 House Prices Dashboard")
# ── KPIs ──────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Precio medio",    f"${df_filtrado['SalePrice'].mean():,.0f}", "#1565c0"),
    (k2, "Precio mediano",  f"${df_filtrado['SalePrice'].median():,.0f}", "#1565c0"),
    (k3, "Precio mínimo",   f"${df_filtrado['SalePrice'].min():,.0f}", "#6a1b9a"),
    (k4, "Precio máximo",   f"${df_filtrado['SalePrice'].max():,.0f}", "#6a1b9a"),
    (k5, "Total casas",     f"{len(df_filtrado):,}", "#2e7d32"),
]
for col, label, val, color in kpis:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{val}</div>
    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Distribución del target
# ══════════════════════════════════════════════════════════════════════════════


st.markdown('<div class="section-title">📊 Distribución de SalePrice</div>',
            unsafe_allow_html=True)

st.plotly_chart(px.histogram(df_filtrado, x="SalePrice"), use_container_width=True) #hISTOGRAMA DE EVOLUCIÓN DE LOS PRECIOS DE VENTA


 



# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Features vs SalePrice
# ══════════════════════════════════════════════════════════════════════════════

  

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Variables categóricas
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Rendimiento del modelo
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Heatmap de correlación
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🔥 Correlaciones</div>',
            unsafe_allow_html=True)
 
num_cols = ["SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF",
            "YearBuilt", "GarageCars", "FullBath", "GarageArea", "1stFlrSF","FullBath","TotRmsAbvGrd"]
corr = df_filtrado[num_cols].corr().round(2)
 
fig12 = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu",
    zmid=0,
    text=corr.values,
    texttemplate="%{text}",
    textfont={"size": 12},
    hoverongaps=False
))
fig12.update_layout(
    title="Matriz de correlación — variables numéricas clave",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig12, use_container_width=True)
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("💡 Los datos de predicción son simulados. Reemplaza `load_data()` con tu `pd.read_csv('train.csv')` y las predicciones de tu modelo entrenado.")