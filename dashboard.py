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
    return pd.read_csv('clean_train.csv')


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
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 Explorador")
    st.markdown("---")
 
    
    variable_categorica = st.selectbox(
        "🎨 Variable de análisis",
        options=["Neighborhood", "BldgType", "HouseStyle", "SaleCondition",
                 "OverallQual", "GarageCars", "FullBath", "YrSold"], #CAMBIAR A LAS SELECCIONADAS EN EL HEADMAP
        help="Esta variable controla el scatter y el pie chart"
    )
 
    st.markdown("---")
    st.markdown("### Filtros")
 
    # Filtro por año de venta — para la evolución histórica
    yr_options = sorted(df["YrSold"].unique())
    yr_sel = st.multiselect(
        "Año de venta",
        options=yr_options,
        default=yr_options
    )
 
    # Filtro por rango de precio
    price_min, price_max = int(df["SalePrice"].min()), int(df["SalePrice"].max())
    price_range = st.slider(
        "Rango de precio ($)",
        price_min, price_max,
        (price_min, price_max),
        step=5000, format="$%d"
    )
 
    # Filtro por calidad general
    qual_range = st.slider("Overall Quality", 1, 10, (1, 10))
 
    # Variable numérica para el eje X del scatter
    st.markdown("---")
    variable_numerica = st.selectbox(
        "📐 Variable eje X del scatter",
        options=["GrLivArea", "LotArea", "TotalBsmtSF", "YearBuilt", "GarageCars"],
        help="Qué variable numérica comparar con SalePrice"
    )
 
 

# ── Filtrado ──────────────────────────────────────────────────────────────────
ddff = df[
    df["YrSold"].isin(yr_sel) &
    df["SalePrice"].between(*price_range) &
    df["OverallQual"].between(*qual_range)
]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏠 House Prices Dashboard")
# ── KPIs ──────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Precio medio",    f"${ddff['SalePrice'].mean():,.0f}", "#1565c0"),
    (k2, "Precio mediano",  f"${ddff['SalePrice'].median():,.0f}", "#1565c0"),
    (k3, "Precio mínimo",   f"${ddff['SalePrice'].min():,.0f}", "#6a1b9a"),
    (k4, "Precio máximo",   f"${ddff['SalePrice'].max():,.0f}", "#6a1b9a"),
    (k5, "Total casas",     f"{len(ddff):,}", "#2e7d32"),
]
for col, label, val, color in kpis:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{val}</div>
    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — EVOLUCIÓN HISTÓRICA DEL PRECIO
# ══════════════════════════════════════════════════════════════════════════════


st.markdown('<div class="section-title">📈 Evolución histórica del precio de venta</div>',
            unsafe_allow_html=True)
 
evol = (ddff.groupby("YrSold")["SalePrice"]
          .agg(media="mean", mediana="median", std="std", n="count")
          .reset_index())
evol["media"]   = evol["media"].round(0)
evol["mediana"] = evol["mediana"].round(0)
evol["upper"]   = (evol["media"] + evol["std"]).clip(upper=price_max)
evol["lower"]   = (evol["media"] - evol["std"]).clip(lower=price_min)
 
fig_evol = go.Figure()
 
# Banda ±1 desviación estándar
fig_evol.add_trace(go.Scatter(
    x=list(evol["YrSold"]) + list(evol["YrSold"])[::-1],
    y=list(evol["upper"])  + list(evol["lower"])[::-1],
    fill="toself", fillcolor="rgba(25, 118, 210, 0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±1 desv. estándar", showlegend=True
))
# Línea de precio medio
fig_evol.add_trace(go.Scatter(
    x=evol["YrSold"], y=evol["media"],
    mode="lines+markers+text",
    line=dict(color="#1976D2", width=3),
    marker=dict(size=9),
    text=[f"${v:,.0f}" for v in evol["media"]],
    textposition="top center",
    textfont=dict(size=11),
    name="Precio medio"
))
# Línea de precio mediano
fig_evol.add_trace(go.Scatter(
    x=evol["YrSold"], y=evol["mediana"],
    mode="lines+markers",
    line=dict(color="#FF7043", width=2, dash="dot"),
    marker=dict(size=7),
    name="Precio mediano"
))
 
fig_evol.update_layout(
    height=380, xaxis_title="Año de venta",
    yaxis_title="Precio ($)", yaxis_tickprefix="$", yaxis_tickformat=",",
    legend=dict(orientation="h", y=1.08),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickmode="linear", dtick=1)
)
st.plotly_chart(fig_evol, use_container_width=True)
 



# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS 2 y 3 — SCATTER + PIE CHART
# ══════════════════════════════════════════════════════════════════════════════

  # ══════════════════════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-title">🔍 Análisis por <i>{variable_categorica}</i></div>',
            unsafe_allow_html=True)
 
col_scatter, col_pie = st.columns([3, 2])
 
with col_scatter:
    # GRÁFICO 2 — SCATTER PLOT precio vs variable numérica seleccionada
    # El color de cada punto viene dado por la variable de análisis del sidebar.
    # Así se puede ver de un vistazo si hay grupos que se comportan diferente.
    # Por ejemplo: seleccionar Neighborhood + GrLivArea en X muestra qué barrios
    # tienen casas grandes y cuáles pequeñas, y si eso se refleja en el precio.
 
    # Si la variable de análisis es numérica, usamos color continuo; si es
    # categórica, usamos paleta cualitativa para distinguir bien los grupos.
    es_categorica = ddff[variable_categorica].dtype == object or \
                    ddff[variable_categorica].nunique() <= 10
 
    if es_categorica:
        fig_scatter = px.scatter(
            ddff, x=variable_numerica, y="SalePrice",
            color=variable_categorica,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"{variable_numerica} vs SalePrice  —  color por {variable_categorica}",
            labels={variable_numerica: variable_numerica,
                    "SalePrice": "Precio ($)",
                    variable_categorica: variable_categorica},
            opacity=0.65, height=420
        )
    else:
        fig_scatter = px.scatter(
            ddff, x=variable_numerica, y="SalePrice",
            color=variable_categorica,
            color_continuous_scale="Viridis",
            title=f"{variable_numerica} vs SalePrice  —  color por {variable_categorica}",
            labels={variable_numerica: variable_numerica,
                    "SalePrice": "Precio ($)"},
            opacity=0.65, height=420
        )
 
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", font=dict(size=11))
    )
    fig_scatter.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_scatter, use_container_width=True)
 
with col_pie:
 
 
    conteo = (ddff[variable_categorica]
              .value_counts()
              .reset_index()
              .rename(columns={"index": variable_categorica,
                               "count": "Casas",
                               variable_categorica: variable_categorica}))
    # value_counts() en pandas >=2.0 devuelve columnas distintas
    if "count" not in conteo.columns:
        conteo.columns = [variable_categorica, "Casas"]
 
    # Agrupamos categorías con menos del 2% en "Otros" para no saturar el gráfico
    total = conteo["Casas"].sum()
    conteo["pct"] = conteo["Casas"] / total
    otros = conteo[conteo["pct"] < 0.02]["Casas"].sum()
    conteo = conteo[conteo["pct"] >= 0.02].copy()
    if otros > 0:
        conteo = pd.concat([
            conteo,
            pd.DataFrame([{variable_categorica: "Otros (<2%)", "Casas": otros, "pct": otros/total}])
        ], ignore_index=True)
 
    fig_pie = px.pie(
        conteo, names=variable_categorica, values="Casas",
        title=f"Distribución de casas por {variable_categorica}",
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.35,   # donut chart — más moderno y legible que pie sólido
        height=420
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent",          # solo % dentro del sector, nombre va en leyenda
        textfont_size=12,
        pull=[0.03] * len(conteo)
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        )
    )
    st.plotly_chart(fig_pie, use_container_width=True)
 
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Variables categóricas
# ════════════════════════════════════


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
corr = ddff[num_cols].corr().round(2)
 
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