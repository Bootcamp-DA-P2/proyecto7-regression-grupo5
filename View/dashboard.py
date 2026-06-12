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
from sklearn.preprocessing import StandardScaler

# ── Configuracion de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="House Prices Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modelo mockeado ────────────────────────────────────────────────────────────
def crear_recursos_mock():
    modelo_mock = LinearRegression()
    scaler_mock = StandardScaler()
    # Entrenamos con un dato ficticio de 7 variables (incluyendo chimeneas)
    X_ficticio = np.array([[7, 1500, 1990, 2, 900, 2, 1]])
    y_ficticio = np.array([12.2]) # Logaritmo aproximado de 200,000
    scaler_mock.fit(X_ficticio)
    # Simulamos los nombres de las columnas en el scaler mock para que no falle
    scaler_mock.feature_names_in_ = ["OverallQual", "GrLivArea", "YearBuilt", "GarageCars", "TotalBsmtSF", "FullBath", "Fireplaces"]
    modelo_mock.fit(scaler_mock.transform(X_ficticio), y_ficticio)
    # Simulamos las 258 columnas coef_ para Ridge
    modelo_mock.coef_ = np.zeros(258)
    # Damos pesos ficticios a los índices que usaremos
    modelo_mock.coef_[0] = 0.35  # OverallQual
    modelo_mock.coef_[1] = 0.25  # GrLivArea
    columnas_mock = ["OverallQual", "GrLivArea", "YearBuilt", "GarageCars", "TotalBsmtSF", "FullBath", "Fireplaces"] + [f"col_{i}" for i in range(251)]
    return modelo_mock, scaler_mock, columnas_mock, True

# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(r'data/utiles/clean_data/clean_train.csv')
df = load_data()

@st.cache_resource
def cargar_recursos():
    try:
        # Cargamos los 3 archivos del pipeline estricto de tu compañera
        modelo = joblib.load(r'data/utiles/modelo/modelo_ridge_house_prices.pkl')
        escalador = joblib.load(r'data/utiles/modelo/escalador_house_prices.pkl')
        columnas = joblib.load(r'data/utiles/modelo/columnas_modelo.pkl')
        return modelo, escalador, columnas, False
    except Exception as e:
        # Si falta algún archivo, el mock salva la app
        return crear_recursos_mock()

# Despaquetamos los 3 componentes del pipeline + el testigo de error
modelo_ml, scaler, columnas_modelo, hubo_error = cargar_recursos()


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

    # ── Controles del PREDICTOR (CAMBIO 1: Añadimos Chimeneas) ────────────────
    else:
        st.markdown("### Características de la casa")
        st.caption("Ajusta los valores para estimar el precio")

        p_qual = st.slider("Calidad general (1-10)", 1, 10, 6)
        p_area = st.slider("Superficie habitable (sqft)", 300, 4000, 1500, step=50)
        p_year = st.slider("Año de construcción", 1870, 2010, 1990)
        p_gar  = st.slider("Plazas de garaje", 0, 4, 2)
        p_bsmt = st.slider("Superficie sótano (sqft)", 0, 3000, 1000, step=50)
        p_bath = st.slider("Baños completos", 0, 4, 2)
        p_fire = st.slider("Chimeneas (Fireplaces)", 0, 4, 1) # <-- Nueva variable


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — EXPLORADOR 
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

    # ── Gráticos 2 y 3: Scatter + Pie ─────────────────────────────────────────
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
# PÁGINA 2 — PREDICCIÓN (CAMBIO 2: Sincronización Rigurosa del Pipeline)
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.title("🔮 Predictor de precio")
    st.caption("Introduce las características de la casa para estimar su valor de venta en tiempo real")
    st.markdown("---")

    if hubo_error:
        st.error("⚠️ No se pudo cargar el modelo real ('mejor_modelo.pkl'). "
                 "La aplicación está funcionando en 'Modo Demostración' con un modelo simulado.")
   
    # 1. Molde de 258 columnas inicializadas en cero
    datos_modelo = {col: 0 for col in columnas_modelo}
    
    # 2. SEGURO: Extraemos exactamente la estructura que el objeto escalador espera
    try:
        columnas_del_scaler = list(scaler.feature_names_in_)
    except AttributeError:
        # Fallback de respaldo ordenado por si falla el atributo nativo
        columnas_del_scaler = ['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'Fireplaces']

    # 3. Inicializamos la tabla del escalador usando sus MEDIAS exactas
    df_escalar = pd.DataFrame([scaler.mean_], columns=columnas_del_scaler)
    
    # 4. Volcamos los datos del Sidebar de tu interfaz
    df_escalar['OverallQual'] = float(p_qual)
    df_escalar['GrLivArea'] = float(p_area)
    df_escalar['YearBuilt'] = float(p_year)
    df_escalar['GarageCars'] = float(p_gar)
    df_escalar['TotalBsmtSF'] = float(p_bsmt)
    df_escalar['FullBath'] = float(p_bath)
    df_escalar['Fireplaces'] = float(p_fire) # Mapeado con tu nuevo slider
    
    # Ajustes lógicos colaterales idénticos a los de ella
    if 'GarageArea' in df_escalar.columns:
        df_escalar['GarageArea'] = float(p_gar * 300)
    if '1stFlrSF' in df_escalar.columns:
        df_escalar['1stFlrSF'] = float(p_bsmt)
        
    # 5. ESCALAMOS (Manteniendo rigurosamente las dimensiones)
    datos_escalados = scaler.transform(df_escalar[columnas_del_scaler])
    df_escalado_limpio = pd.DataFrame(datos_escalados, columns=columnas_del_scaler)
    
    # 6. Pasamos los valores numéricos escalados al diccionario final del modelo
    for col in df_escalado_limpio.columns:
        if col in datos_modelo:
            datos_modelo[col] = df_escalado_limpio.loc[0, col]
            
    # 7. Construimos el DataFrame definitivo estructurado con las 258 columnas exactas
    df_final_scoring = pd.DataFrame([datos_modelo], columns=columnas_modelo)
    
    # 8. Predicción final y reversión del logaritmo (np.expm1)
    prediccion_log = modelo_ml.predict(df_final_scoring)
    pred = float(np.expm1(prediccion_log[0]))
    
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
    st.markdown('<div class="section-title">📊 Explicación de la Predicción y Comparativa</div>',
                unsafe_allow_html=True)

    col_pred_1, col_pred_2 = st.columns(2)

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICO A: Contribución Real (CAMBIO 3: Coeficientes de Ridge en Plotly)
    # ──────────────────────────────────────────────────────────────────────────
    with col_pred_1:
        variables_clave = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'Fireplaces']
        features_nombres = ["Calidad", "Superficie (sqft)", "Año Const.", "Sótano (sqft)", "Plazas Garaje", "Baños", "Chimeneas"]
        
        try:
            indices_columnas = [columnas_modelo.index(c) for c in variables_clave]
            pesos = modelo_ml.coef_[indices_columnas]
        except (AttributeError, ValueError):
            pesos = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05, 0.03]
        
        df_imp = pd.DataFrame({
            'Característica': features_nombres,
            'Fuerza del Coeficiente': pesos
        }).sort_values(by='Fuerza del Coeficiente', ascending=True)

        df_imp['Efecto'] = ['Sube el Precio' if x >= 0 else 'Baja el Precio' for x in df_imp['Fuerza del Coeficiente']]

        fig_contrib = px.bar(
            df_imp, 
            x='Fuerza del Coeficiente', 
            y='Característica',
            orientation='h',
            title="¿Qué características influyen más en el modelo Ridge?",
            color='Efecto',
            color_discrete_map={'Sube el Precio': '#1E88E5', 'Baja el Precio': '#FF4B4B'}
        )
        
        fig_contrib.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2), height=350
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICO B: Comparativa con Casas Similares (Se mantiene intacto y funcional)
    # ──────────────────────────────────────────────────────────────────────────
    with col_pred_2:
        casas_similares = df[
            df["OverallQual"].between(p_qual - 1, p_qual + 1) &
            df["GrLivArea"].between(p_area - 300, p_area + 300)
        ].head(5)

        if not casas_similares.empty:
            casas_similares = casas_similares.sort_values(by="SalePrice")
            casas_similares["ID_Casa"] = [f"Casa Real {i+1}" for i in range(len(casas_similares))]
            
            casa_usuario = pd.DataFrame([{
                "ID_Casa": "⭐ TU PREDICCIÓN",
                "SalePrice": pred
            }])
            
            df_comp = pd.concat([casas_similares[["ID_Casa", "SalePrice"]], casa_usuario], ignore_index=True)
            colores = ["#6c757d"] * (len(df_comp) - 1) + ["#1565c0"]

            fig_comp = go.Figure(go.Bar(
                x=df_comp["ID_Casa"],
                y=df_comp["SalePrice"],
                marker_color=colores,
                text=[f"${val:,.0f}" for val in df_comp["SalePrice"]],
                textposition='auto'
            ))
            
            fig_comp.update_layout(
                title=f"Tu estimación vs casas reales (Calidad ~{p_qual} y Área ~{p_area} sqft)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350
            )
            fig_comp.update_yaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig_comp, use_container_width=True)

    
        else:
            st.warning("⚠️ No se encontraron casas reales lo suficientemente similares para comparar en este rango.")
    
    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICO C: Distribución del Mercado (Campana de Gauss / Histograma)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏠 Posición de tu casa en la distribución del mercado</div>',
                unsafe_allow_html=True)

    # Usamos los precios reales del dataset para crear la distribución
    precios_reales = df["SalePrice"]

    fig_dist = go.Figure()

    # 1. Creamos el histograma (la campana) con los datos reales del CSV
    fig_dist.add_trace(go.Histogram(
        x=precios_reales,
        name='Distribución del Mercado',
        marker_color='teal',
        opacity=0.5,
        nbinsx=50,
        histnorm='probability density' # Esto hace que parezca una campana de densidad
    ))

    # 2. Añadimos la línea vertical roja que indica la PREDICCIÓN actual
    fig_dist.add_vline(
        x=pred, 
        line_width=4, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f" TU PREDICCIÓN: ${pred:,.0f}",
        annotation_position="top right",
        annotation_font_color="red"
    )

    fig_dist.update_layout(
        title="¿Es tu casa barata o cara respecto al total del mercado?",
        xaxis_title="Precio de Venta ($ USD)",
        yaxis_title="Densidad de Propiedades",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        showlegend=False,
        # Limitamos el eje X para que se vea bien la parte central del mercado
        xaxis=dict(range=[0, 500000], tickprefix="$", tickformat=",")
    )

    st.plotly_chart(fig_dist, use_container_width=True)