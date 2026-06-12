import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
import joblib # Para cargar el modelo de machine learning
from sklearn.linear_model import LinearRegression #creación de modeo mockeado

# ── Configuracion de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="House Prices Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modelo mockeado ────────────────────────────────────────────────────────────
def crear_modelo_mock():
    modelo_mock = LinearRegression()
    # Supone 6 variables de entrada (p_qual, p_area, p_year, p_gar, p_bsmt, p_bath)
    X_ficticio = np.array([[7, 1500, 1990, 2, 900, 2]])
    y_ficticio = np.array([200000.0])
    modelo_mock.fit(X_ficticio, y_ficticio)
    return modelo_mock
# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv('clean_train.csv')
df = load_data()

@st.cache_resource  
def cargar_modelo():
    try:
        return joblib.load("mejor_modelo.pkl"), False
    except Exception as e:
        # Si no encuentra 'mejor_modelo.pkl', carga el mock
        modelo_respaldo = crear_modelo_mock()
        return modelo_respaldo, True


modelo_ml, hubo_error = cargar_modelo()




# ── Estilos ───────────────────────────────────────────────────────────────────
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
    .pred-result {
        background: linear-gradient(135deg, #1565c0 0%, #4a6cf7 100%);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-result .label { font-size: 14px; opacity: 0.85; margin-bottom: 6px; }
    .pred-result .value { font-size: 42px; font-weight: 700; letter-spacing: -1px; }
    .pred-result .range { font-size: 13px; opacity: 0.7; margin-top: 6px; }
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { gap: 8px; }
    div[data-testid="stRadio"] > div > label {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 10px 14px;
        width: 100%;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        border: 1.5px solid transparent;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background: #e2e8f5;
        border-color: #4a6cf7;
    }
    [data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 Explorador")
    st.markdown("---")

    pagina = st.radio(
        "Navegación",
        options=["📊  Explorador", "🔮  Predicción"],
        index=0
    )

    st.markdown("---")

    # ── Controles del EXPLORADOR ──────────────────────────────────────────────
    if pagina == "📊  Explorador":

        variable_categorica = st.selectbox(
            "🎨 Variable de análisis",
            options=["Neighborhood", "BldgType", "HouseStyle", "SaleCondition",
                     "OverallQual", "GarageCars", "FullBath", "YrSold"],
            help="Esta variable controla el scatter y el pie chart"
        )

        st.markdown("---")
        st.markdown("### Filtros")

        yr_options = sorted(df["YrSold"].unique())
        yr_sel = st.multiselect(
            "Año de venta",
            options=yr_options,
            default=yr_options
        )

        price_min, price_max = int(df["SalePrice"].min()), int(df["SalePrice"].max())
        price_range = st.slider(
            "Rango de precio ($)",
            price_min, price_max,
            (price_min, price_max),
            step=5000, format="$%d"
        )

        qual_range = st.slider("Overall Quality", 1, 10, (1, 10))

        st.markdown("---")
        variable_numerica = st.selectbox(
            "📐 Variable eje X del scatter",
            options=["GrLivArea", "LotArea", "TotalBsmtSF", "YearBuilt", "GarageCars"],
            help="Qué variable numérica comparar con SalePrice"
        )

    # ── Controles del PREDICTOR ───────────────────────────────────────────────
    else:
        st.markdown("### Características de la casa")
        st.caption("Ajusta los valores para estimar el precio")

        p_qual = st.slider("Calidad general (1-10)", 1, 10, 7)
        p_area = st.slider("Superficie habitable (sqft)", 500, 4000, 1500, step=50)
        p_year = st.slider("Año de construcción", 1900, 2010, 1990)
        p_gar  = st.slider("Plazas de garaje", 0, 4, 2)
        p_bsmt = st.slider("Superficie sótano (sqft)", 0, 3000, 900, step=50)
        p_bath = st.slider("Baños completos", 0, 3, 2)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — EXPLORADOR (tu código original intacto)
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📊  Explorador":

    ddff = df[
        df["YrSold"].isin(yr_sel) &
        df["SalePrice"].between(*price_range) &
        df["OverallQual"].between(*qual_range)
    ]

    st.title("🏠 House Prices Dashboard")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "Precio medio",   f"${ddff['SalePrice'].mean():,.0f}",   "#1565c0"),
        (k2, "Precio mediano", f"${ddff['SalePrice'].median():,.0f}", "#1565c0"),
        (k3, "Precio mínimo",  f"${ddff['SalePrice'].min():,.0f}",    "#6a1b9a"),
        (k4, "Precio máximo",  f"${ddff['SalePrice'].max():,.0f}",    "#6a1b9a"),
        (k5, "Total casas",    f"{len(ddff):,}",                      "#2e7d32"),
    ]
    for col, label, val, color in kpis:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gráfico 1: Evolución histórica ────────────────────────────────────────
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
    fig_evol.add_trace(go.Scatter(
        x=list(evol["YrSold"]) + list(evol["YrSold"])[::-1],
        y=list(evol["upper"])  + list(evol["lower"])[::-1],
        fill="toself", fillcolor="rgba(25, 118, 210, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 desv. estándar", showlegend=True
    ))
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

    # ── Gráficos 2 y 3: Scatter + Pie ─────────────────────────────────────────
    st.markdown(f'<div class="section-title">🔍 Análisis por <i>{variable_categorica}</i></div>',
                unsafe_allow_html=True)

    col_scatter, col_pie = st.columns([3, 2])

    with col_scatter:
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
        if "count" not in conteo.columns:
            conteo.columns = [variable_categorica, "Casas"]

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
            hole=0.35, height=420
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent",
            textfont_size=12, pull=[0.03] * len(conteo)
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                orientation="v", x=1.02, y=0.5,
                xanchor="left", yanchor="middle",
                font=dict(size=12), bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Heatmap de correlación ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔥 Correlaciones</div>',
                unsafe_allow_html=True)

    num_cols = ["SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF",
                "YearBuilt", "GarageCars", "FullBath", "GarageArea",
                "1stFlrSF", "TotRmsAbvGrd"]
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


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — PREDICCIÓN (casilla lista, gráficos por decidir)
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.title("🔮 Predictor de precio")
    st.caption("Introduce las características de la casa para estimar su valor de venta")
    st.markdown("---")

    if hubo_error:
        st.error("⚠️ No se pudo cargar el modelo real ('mejor_modelo.pkl'). "
                 "La aplicación está funcionando en 'Modo Demostración' con un modelo simulado.")
   
    datos_casa = np.array([[p_qual, p_area, p_year, p_gar, p_bsmt, p_bath]])
    
    # 2. Hacer la predicción real con el .pkl
    pred_real = modelo_ml.predict(datos_casa)
    pred = float(pred_real[0]) # Extraemos el número del array de predicción
    
    # Mantener tu cálculo del margen estético del 8%
    margen = float(pred * 0.08)

    st.markdown(f"""
    <div class="pred-result">
        <div class="label">Precio estimado</div>
        <div class="value">${pred:,.2f}</div>
        <div class="range">Rango estimado: ${pred-margen:,.2f} — ${pred+margen:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # AQUÍ VAN LOS GRÁFICOS DE PREDICCIÓN — por decidir con el equipo
    # ==========================================================================

    # Gráfico A — ej. contribución de cada feature
    # Gráfico B — ej. comparativa con casas similares

    # ==========================================================================


