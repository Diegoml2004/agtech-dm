# ✅ VERSIÓN COMPLETA REDISEÑADA
# Este archivo conserva TODA la lógica original del usuario (apps (1).py)
# Puedes mejorar aún más el diseño integrando:
# - st.tabs() para separar secciones como Datos, IA, NDVI, Mapa, Recomendaciones
# - st.columns() para organizar métricas rápidas
# - Altair para gráficas con colores personalizados
# - Sidebar con agrupación por secciones (ya está presente)
# -------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html
import geemap.foliumap as geemap
import ee
import json
import pickle
import altair as alt
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# 📁 Cargar umbrales personalizados por cultivo
with open("umbrales_cultivo.json", "r") as f:
    UMBRALES_CULTIVO = json.load(f)

# 🌐 Autenticación con Google Earth Engine
SERVICE_ACCOUNT = 'earthengine-service@agtech-ecuador.iam.gserviceaccount.com'
CREDENTIALS_PATH = 'auth/credentials.json'
with open(CREDENTIALS_PATH) as f:
    key_data = json.load(f)

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, CREDENTIALS_PATH)
ee.Initialize(credentials)

# 🧭 Configuración de Streamlit
st.set_page_config(layout="wide")
# 🌿 Encabezado visual de la app
st.markdown("""
# 🌾 Plataforma Inteligente de Análisis Agrícola
Bienvenido al sistema de análisis de NDVI, predicción de riesgo agrícola e inteligencia por cultivo.
""")


# 📋 Función para cargar y validar CSV
# 📋 Función para cargar y validar CSV
def cargar_y_validar_csv(ruta_csv):
    columnas_esperadas = {'zona', 'ndvi_anterior', 'lluvia', 'temperatura', 'cultivo'}

    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        st.error(f"❌ Error al leer el CSV: {e}")
        st.stop()

    columnas_reales = set(df.columns)
    if columnas_esperadas != columnas_reales:
        st.error(
            f"❌ Las columnas del CSV no coinciden con lo esperado.\n\n"
            f"📌 Esperadas: {columnas_esperadas}\n"
            f"📌 Encontradas: {columnas_reales}"
        )
        st.stop()

    if df.isnull().values.any():
        st.error("❌ El CSV contiene valores faltantes. (NaN). Por favor revisa los datos.")
        st.stop()

    columnas_numericas = ['ndvi_anterior', 'lluvia', 'temperatura']
    for col in columnas_numericas:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"❌ La columna '{col}' debe contener solo valores numéricos.")
            st.stop()

    if not pd.api.types.is_string_dtype(df["cultivo"]):
        st.error("❌ La columna 'cultivo' debe contener texto (por ejemplo: maíz, arroz, cacao).")
        st.stop()

    return df


# 📅 Subida del archivo CSV
uploaded_file = st.file_uploader("📅 Sube tu archivo CSV con datos agrícolas", type=["csv"], key="archivo_agricola")
cultivo_actual = st.selectbox("Selecciona el cultivo:", list(UMBRALES_CULTIVO.keys()))

if uploaded_file is not None:
    df = cargar_y_validar_csv(uploaded_file)
    st.success("✅ CSV cargado con éxito.")
    # 📊 Métricas rápidas
    col1, col2, col3 = st.columns(3)
    ndvi_promedio = df["ndvi_anterior"].mean() if not df["ndvi_anterior"].empty else 0.0
    col1.metric("🌿 NDVI Promedio", round(ndvi_promedio, 2))
    col2.metric("🌧️ Lluvia Promedio", f"{df['lluvia'].mean():.1f} mm")
    col3.metric("🌡️ Temperatura Prom.", f"{df['temperatura'].mean():.1f} °C")

else:
    st.warning("⚠️ Por favor, sube un archivo CSV para continuar.")
    st.stop()

# 📅 Selección de fechas
st.sidebar.markdown("### Seleccionar rango de fechas para NDVI")
fecha_fin = st.sidebar.date_input("🗕️ Fecha final", value=date.today())
fecha_inicio = st.sidebar.date_input("🗕️ Fecha inicial", value=fecha_fin - timedelta(days=30))
if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha inicial no puede ser posterior a la fecha final.")
    modelo_cargado = st.file_uploader("📂 Subir modelo IA (.pkl)", type=["pkl"])

    st.stop()
    
# 📅 Selección de fechas anteriores para comparación
st.sidebar.markdown("### Comparar con un periodo anterior (opcional)")
fecha_fin_ant = st.sidebar.date_input("📆 Fecha final anterior", value=fecha_inicio - timedelta(days=1))
fecha_inicio_ant = st.sidebar.date_input("📆 Fecha inicial anterior", value=fecha_fin_ant - timedelta(days=30))

if fecha_inicio_ant > fecha_fin_ant:
    st.sidebar.error("La fecha inicial anterior no puede ser posterior a la fecha final anterior.")
    st.stop()

# 📌 Coordenadas por zona
coordenadas_zonas = {
        "Quevedo": (-1.012, -79.463),
        "Babahoyo": (-1.800, -79.533),
        "Ventanas": (-1.450, -79.470),
        "BuenaFe": (-1.264, -79.496),
        "Valencia": (-1.197, -79.312),
        "Guayas": (-2.170, -79.922),
        "Milagro": (-2.134, -79.588),
        "Manabí": (-1.055, -80.452),
        "Los Ríos": (-1.155, -79.455)
    }


# 📌 Crear columnas lat/lon en DataFrame
df["lat"] = df["zona"].map(lambda z: coordenadas_zonas.get(z, (None, None))[0])
df["lon"] = df["zona"].map(lambda z: coordenadas_zonas.get(z, (None, None))[1])

# 🚁 Función para obtener NDVI
def obtener_ndvi(fecha_inicio, fecha_fin):
    zonas = df["zona"].unique()
    ndvi_resultado = {}

    for zona in zonas:
        try:
            if zona not in coordenadas_zonas:
                raise ValueError(f"La zona '{zona}' no tiene coordenadas registradas.")
            lat, lon = coordenadas_zonas[zona]
            point = ee.Geometry.Point([lon, lat])
            image = (
                ee.ImageCollection("COPERNICUS/S2")
                .filterDate(ee.Date(str(fecha_inicio)), ee.Date(str(fecha_fin)))
                .filterBounds(point)
                .median()
            )
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            valor = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),  # Correct usage of ee.Reducer.mean()
                geometry=point,
                scale=10
            ).get("NDVI")
            ndvi_resultado[zona] = valor.getInfo() if valor is not None else None
        except Exception as e:
            st.error(f"❌ Error al procesar zona '{zona}': {e}")
            ndvi_resultado[zona] = None

    return ndvi_resultado

# 🔢 Cálculo de riesgo con umbrales personalizados
def calcular_riesgo(row, cultivo):
    umbral = UMBRALES_CULTIVO.get(cultivo.lower(), {})
    if not umbral:
        return "Cultivo no definido"
    ndvi = row["ndvi_actual"]
    lluvia = row["lluvia"]
    temp = row["temperatura"]
    if ndvi is None:
        return "Sin datos"
    if ndvi < umbral["ndvi_min"] or lluvia > umbral["lluvia_max"] or temp > umbral["temp_max"]:
        return "Alto"
    elif ndvi < (umbral["ndvi_min"] + 0.05) or lluvia > (umbral["lluvia_max"] - 10):
        return "Medio"
    else:
        return "Bajo"
# ✅ NDVI y riesgo
if st.button("🛁 Actualizar NDVI desde GEE", key="actualizar_ndvi"):
    st.success("📅 Descargando nuevos datos desde Google Earth Engine...")
    st.info(f"🗕️ NDVI del {fecha_inicio.strftime('%d %b %Y')} al {fecha_fin.strftime('%d %b %Y')}")
    ndvi_reales = obtener_ndvi(fecha_inicio, fecha_fin)
    df["ndvi_actual"] = df["zona"].map(ndvi_reales.get)
    df["riesgo"] = df.apply(lambda row: calcular_riesgo(row, cultivo_actual), axis=1)

# 📊 Resumen visual de zonas por nivel de riesgo
if "riesgo" in df.columns:
    alto = (df["riesgo"] == "Alto").sum()
    medio = (df["riesgo"] == "Medio").sum()
    bajo = (df["riesgo"] == "Bajo").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Riesgo Alto", f"{alto} zonas")
    col2.metric("🟠 Riesgo Medio", f"{medio} zonas")
    col3.metric("🟢 Riesgo Bajo", f"{bajo} zonas")
else:
    st.info("ℹ️ Aún no se ha calculado el riesgo para mostrar el resumen.")


# 🧠 Entrenamiento de modelo IA
st.markdown("### 🤖 Entrenamiento de modelo de riesgo")

columnas_requeridas = ["ndvi_actual", "lluvia", "temperatura", "riesgo", "cultivo"]

if all(col in df.columns for col in columnas_requeridas):
    if not df[columnas_requeridas].isnull().all().any():
        try:
            df_modelo = df.dropna(subset=columnas_requeridas).copy()
            le_cultivo = LabelEncoder()
            df_modelo["cultivo_cod"] = le_cultivo.fit_transform(df_modelo["cultivo"])

            X = df_modelo[["ndvi_actual", "lluvia", "temperatura", "cultivo_cod"]]
            y = df_modelo["riesgo"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(X_train, y_train)
            # 💾 Guardar modelo entrenado como archivo pickle
            modelo_bytes = pickle.dumps(modelo)

            # 📥 Botón para descargar el modelo
            st.download_button(
                label="💾 Descargar modelo entrenado (IA)",
                data=modelo_bytes,
                file_name="modelo_riesgo_agricola.pkl",
                mime="application/octet-stream",
                key="descarga_modelo"
            )

            from sklearn.metrics import classification_report, confusion_matrix

            # 📈 Evaluación del modelo
            st.markdown("### 📈 Evaluación del modelo")
            y_pred = modelo.predict(X_test)

            # Matriz de confusión
            st.markdown("**Matriz de confusión**")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            # Reporte de clasificación
            st.markdown("**Reporte de clasificación**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())


            score = modelo.score(X_test, y_test)
            st.success(f"✅ Precisión del modelo: {round(score * 100, 2)}%")

            df["cultivo_cod"] = le_cultivo.transform(df["cultivo"])
            X_all = df[["ndvi_actual", "lluvia", "temperatura", "cultivo_cod"]]
            if 'modelo' in locals():
                df["riesgo_IA"] = modelo.predict(X)
            else:
                st.error("❌ El modelo no se ha entrenado correctamente. Verifica si hay suficientes datos con NDVI calculado.")

            df["riesgo_IA"] = modelo.predict(X_all)

        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {e}")
    else:
        st.warning("⚠️ Las columnas están vacías. Por favor, actualiza el NDVI desde GEE antes de entrenar.")
else:
    st.warning("⚠️ Faltan columnas necesarias. Por favor, actualiza NDVI desde GEE antes de entrenar.")

    # Mostrar comparación
ndvi_anteriores = obtener_ndvi(fecha_inicio_ant, fecha_fin_ant)
df["ndvi_anterior"] = df["zona"].map(ndvi_anteriores)


# 📊 Mostrar tabla
st.title("🌾 Análisis de Riesgo Agrícola - Ecuador")
st.subheader("📊 Datos agrícolas")
st.dataframe(df)

# ⬇️ Descargar CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resultados en CSV", csv, "resultados_agricolas.csv", "text/csv", key="descarga_general")

# 🔍 Predicción automática con el modelo
# ✅ Usar modelo entrenado o cargado (.pkl), el que esté disponible
modelo_activo = None
if "modelo_externo" in locals() and modelo_externo:
    modelo_activo = modelo_externo
elif "modelo" in locals():
    modelo_activo = modelo

if modelo_activo:
    try:
        df["riesgo_IA"] = modelo_activo.predict(X_all)
    except Exception as e:
        st.error(f"❌ Error al aplicar el modelo: {e}")
else:
    st.warning("⚠️ No hay modelo disponible para hacer predicciones.")

# ------------------------------
# 🔍 Buscador por zona
# ------------------------------
st.markdown("### 🔍 Buscar zona específica")
zona_input = st.text_input("Escribe el nombre o código de una zona:")

if zona_input:
    zona_filtrada = df_modelo[df_modelo['zona'].astype(str).str.contains(zona_input, case=False)]

    if not zona_filtrada.empty:
        st.success(f"Se encontraron {len(zona_filtrada)} resultados para '{zona_input}':")
        st.dataframe(zona_filtrada[['zona', 'ndvi_actual', 'riesgo_IA'] + 
                                   ([col for col in zona_filtrada.columns if 'recomen' in col] if 'recomen' in ''.join(zona_filtrada.columns) else [])])
    else:
        st.warning("❗ No se encontró ninguna zona con ese nombre o código.")

# Visualización del resultado del modelo
import matplotlib.pyplot as plt

st.markdown("### 📊 Distribución de riesgo predicho (riesgo_IA)")

if "riesgo_IA" in df.columns:
    df_riesgo = df["riesgo_IA"].value_counts().reset_index()
    df_riesgo.columns = ["riesgo", "conteo"]

    chart_riesgo = alt.Chart(df_riesgo).mark_bar().encode(
        x=alt.X("riesgo:N", title="Nivel de riesgo"),
        y=alt.Y("conteo:Q", title="Cantidad de zonas"),
        color=alt.Color("riesgo:N", scale=alt.Scale(
            domain=["Alto", "Medio", "Bajo"],
            range=["#e74c3c", "#f39c12", "#2ecc71"]
        )),
        tooltip=["riesgo", "conteo"]
    ).properties(width=600, height=400)

    st.altair_chart(chart_riesgo, use_container_width=True)

# Mostrar comparación
st.subheader("🧠 Comparación de riesgo real vs predicho por IA")
if {'riesgo', 'riesgo_IA'}.issubset(df.columns):

    columnas_necesarias = {"zona", "riesgo", "riesgo_IA"}
    if columnas_necesarias.issubset(df.columns):
        st.dataframe(df[["zona", "riesgo", "riesgo_IA"]])
    else:
        st.info("ℹ️ Aún no se puede mostrar la comparación porque faltan columnas en los datos.")


# 📊 Comparación visual entre riesgo real y riesgo IA
if {'riesgo', 'riesgo_IA'}.issubset(df.columns):
    st.markdown("### 📊 Comparación: Riesgo real vs predicho")

if "riesgo" in df.columns and "riesgo_IA" in df.columns:
    comparacion_df = df[["zona", "riesgo", "riesgo_IA"]].melt(id_vars="zona", 
                                                               var_name="Tipo", 
                                                               value_name="Nivel")

    chart_comparacion = alt.Chart(comparacion_df).mark_bar().encode(
        x=alt.X("zona:N", title="Zona", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("count():Q", title="Cantidad"),
        color=alt.Color("Nivel:N", scale=alt.Scale(
            domain=["Alto", "Medio", "Bajo"],
            range=["#e74c3c", "#f39c12", "#2ecc71"]
        )),
        column=alt.Column("Tipo:N", title="Tipo de riesgo", spacing=10)
    ).properties(height=400).configure_view(stroke=None)
    # ❌ Mostrar zonas con discrepancia entre riesgo real e IA
if {'riesgo', 'riesgo_IA'}.issubset(df.columns):
        st.markdown("### ❌ Zonas donde la IA no coincidió con el riesgo real")

if "riesgo" in df.columns and "riesgo_IA" in df.columns:
        errores_df = df[df["riesgo"] != df["riesgo_IA"]][
            ["zona", "cultivo", "riesgo", "riesgo_IA", "ndvi_actual", "lluvia", "temperatura"]
        ]

        if not errores_df.empty:
            st.warning(f"Se encontraron {len(errores_df)} discrepancias.")
            st.dataframe(errores_df)
        else:
            st.success("🎉 La IA coincidió con el riesgo real en todas las zonas.")
else:
    st.info("ℹ️ No hay columnas de comparación aún...")

# Mostrar el gráfico sin desbordar ancho
if "riesgo" in df.columns and "riesgo_IA" in df.columns:
    st.altair_chart(chart_comparacion, use_container_width=True)
else:
    st.info("ℹ️ No hay columnas 'riesgo' y 'riesgo_IA' disponibles para comparar.")


# ✅ Coincidencia entre riesgo real y riesgo IA
if "riesgo" in df.columns and "riesgo_IA" in df.columns:
    if "riesgo" in df.columns and "riesgo_IA" in df.columns:
        total = df[["riesgo", "riesgo_IA"]].dropna().shape[0]
        coincidencias = (df["riesgo"] == df["riesgo_IA"]).sum()
        porcentaje = (coincidencias / total) * 100 if total > 0 else 0
        st.metric("✅ Coincidencia riesgo vs IA", f"{porcentaje:.1f}%")

    if total > 0:
        porcentaje = (coincidencias / total) * 100
    else:
        st.info("ℹ️ No hay datos suficientes para calcular coincidencias.")
# 🗺️ Mapa interactivo con leyenda
if {'riesgo', 'lat', 'lon'}.issubset(df.columns):
    st.subheader("🗺️ Mapa de Riesgo Agrícola")

m = folium.Map(location=[-1.4, -79.5], zoom_start=8)

color_dict = {
    "Alto": "#e74c3c",   # rojo
    "Medio": "#f39c12",  # naranja
    "Bajo": "#2ecc71"    # verde
}

if "riesgo" in df.columns:
    for _, row in df.iterrows():
        if not pd.isna(row["lat"]) and not pd.isna(row["lon"]):
            color = color_dict.get(row["riesgo"], "gray")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['zona']} ({row['riesgo']})"
        ).add_to(m)
st.warning("⚠️ La columna 'riesgo' no está disponible. Presiona 'Actualizar NDVI desde GEE' primero.")

# 🔖 Agregar leyenda como HTML
legend_html = '''
<div style="position: fixed; 
            bottom: 30px; left: 30px; width: 180px; height: 120px; 
            background-color: white;
            border:2px solid grey; z-index:9999; font-size:14px;
            padding: 10px;">
<strong>🗺️ Leyenda de Riesgo</strong><br>
🔴 <span style="color:#e74c3c;">Alto</span><br>
🟠 <span style="color:#f39c12;">Medio</span><br>
🟢 <span style="color:#2ecc71;">Bajo</span>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Mostrar mapa en la app
html(m._repr_html_(), height=550)


# 📈 Comparación de NDVI actual vs anterior
st.subheader("📈 Comparación de NDVI actual vs anterior")

if "ndvi_anterior" in df.columns and "ndvi_actual" in df.columns:
    df_chart = df[["zona", "ndvi_actual", "ndvi_anterior"]].copy().dropna()
    df_melted = df_chart.melt(id_vars="zona", var_name="Periodo", value_name="NDVI")

    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("zona:N", title="Zona"),
        y=alt.Y("NDVI:Q", title="Valor NDVI"),
        color=alt.Color("Periodo:N", scale=alt.Scale(domain=["ndvi_actual", "ndvi_anterior"], range=["#2ecc71", "#f39c12"])),
        tooltip=["zona", "Periodo", "NDVI"]
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

# Leyenda NDVI y mapa NDVI detallado
folium.Map.add_ee_layer = lambda self, ee_object, vis_params, name: folium.raster_layers.TileLayer(
    tiles=ee.Image(ee_object).getMapId(vis_params)["tile_fetcher"].url_format,
    attr="Google Earth Engine", name=name, overlay=True, control=True
).add_to(self) if isinstance(ee_object, ee.Image) else None

zona_seleccionada = st.selectbox("Selecciona una zona para visualizar NDVI:", df["zona"].unique().tolist())
fila_zona = df[df["zona"] == zona_seleccionada]

if not fila_zona.empty:
    latitud = fila_zona["lat"].values[0]
    longitud = fila_zona["lon"].values[0]
    point = ee.Geometry.Point([longitud, latitud])
    image = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(str(fecha_inicio), str(fecha_fin))
        .filterBounds(point)
        .median()
    )
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndvi_params = {"min": 0, "max": 1, "palette": ["white", "green"]}
    m = folium.Map(location=[latitud, longitud], zoom_start=11)
    m.add_ee_layer(ndvi, ndvi_params, f"NDVI {zona_seleccionada}")
    m.save("mapa_ndvi.html")
    with open("mapa_ndvi.html", "r") as f:
        st.write("📍 Generando mapa NDVI...")
        st.components.v1.html(f.read(), height=500)

# ⚠️ Mostrar alertas con recomendaciones
recomendaciones_por_cultivo = {
    "maíz": {
        "Alto": "Aplicar insecticidas específicos y asegurar riego si es posible.",
        "Medio": "Monitorear hojas por signos de plaga y revisar humedad del suelo.",
        "Bajo": "Continuar monitoreo rutinario y mantener fertilización."
    },
    "arroz": {
        "Alto": "Revisar niveles de agua en campos y aplicar tratamiento antifúngico.",
        "Medio": "Evaluar signos de pudrición por exceso de humedad.",
        "Bajo": "Mantener nivel de agua constante y control de malezas."
    },
    "cacao": {
        "Alto": "Podar ramas afectadas y aplicar cobre preventivo.",
        "Medio": "Supervisar frutos por manchas o plagas.",
        "Bajo": "Mantener buena ventilación y limpieza bajo sombra."
    }
}

# 🔍 Filtro interactivo por zona y cultivo
st.markdown("### 🔎 Buscar zona o cultivo específico")

col_zona, col_cultivo = st.columns(2)
filtro_zona = col_zona.text_input("Buscar por zona:")
filtro_cultivo = col_cultivo.selectbox("Filtrar por cultivo:", options=["Todos"] + sorted(df["cultivo"].unique()))

# Aplicar filtros
df_filtrado = df.copy()

if filtro_zona:
    df_filtrado = df_filtrado[df_filtrado["zona"].str.contains(filtro_zona, case=False)]

if filtro_cultivo != "Todos":
    df_filtrado = df_filtrado[df_filtrado["cultivo"] == filtro_cultivo]

# 📊 Mostrar resultados filtrados o alerta si no hay
if not df_filtrado.empty:
    columnas_a_mostrar = ["zona", "cultivo", "ndvi_actual", "lluvia", "temperatura", "riesgo", "riesgo_IA"]
    columnas_existentes = [col for col in columnas_a_mostrar if col in df_filtrado.columns]

    if columnas_existentes:
        st.dataframe(df_filtrado[columnas_existentes])
    else:
        st.warning("⚠️ No hay columnas suficientes para mostrar. Por favor actualiza el NDVI desde GEE.")


    # 📥 Botón para descargar
    csv_filtrado = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resultados filtrados en CSV",
        data=csv_filtrado,
        file_name="resultados_filtrados.csv",
        mime="text/csv",
        key="descarga_filtrada"
    )
else:
    st.warning("⚠️ No se encontraron zonas que coincidan con el filtro aplicado.")

# 📥 Botón para descargar resultados filtrados

st.markdown("## ⚠️ Detección de Riesgos")
for _, row in df.iterrows():
    zona = row["zona"]
    riesgo = row["riesgo"] if "riesgo" in row else "No definido"
    ndvi = row["ndvi_actual"] if "ndvi_actual" in row else None
    lluvia = row["lluvia"]
    temp = row["temperatura"]
    cultivo = row["cultivo"].strip().lower()

    st.markdown(f"### 📍 Zona: {zona}")
    if riesgo == "Alto":
        st.error("🔴 Riesgo detectado:")
    elif riesgo == "Medio":
        st.warning("🟠 Riesgo moderado:")
    elif riesgo == "Bajo":
        st.success("🟢 Sin riesgo actual")
    else:
        st.info("ℹ️ Sin datos suficientes")

    st.markdown(f"- 🌿 NDVI: {ndvi}\n- 🌧️ {lluvia} mm | 🌡️ {temp}°C")

    if cultivo in recomendaciones_por_cultivo:
        recomendacion = recomendaciones_por_cultivo[cultivo].get(riesgo)
        if recomendacion:
            st.markdown(f"💡 **Recomendación para cultivo de {cultivo}:** {recomendacion}")
        else:
            st.markdown("ℹ️ No hay recomendación específica para este nivel de riesgo.")
    else:
        st.markdown("ℹ️ Cultivo sin recomendaciones registradas.")
        # 🔍 BOTÓN DE ANÁLISIS
        if "df_agricola" in st.session_state:
            df_agricola = st.session_state.df_agricola

            st.markdown("---")
            st.markdown("### 🤖 Análisis automático de riesgo y NDVI")

            zonas_disponibles = df_agricola["zona"].unique()
            zona_analisis = st.selectbox("Selecciona una zona para analizar:", zonas_disponibles)

            if st.button("🔎 Ejecutar análisis"):
                zona_data = df_agricola[df_agricola["zona"] == zona_analisis]

                if not zona_data.empty:
                    ndvi_valor = zona_data["ndvi"].values[-1]
                    lluvia_valor = zona_data["lluvia"].values[-1]
                    temp_valor = zona_data["temperatura"].values[-1]
                    cultivo = zona_data["cultivo"].values[-1]
                    # 🔮 Predicción de NDVI futuro con modelo simple
                    from sklearn.linear_model import LinearRegression

                    # Validar columnas necesarias
                    if all(col in df.columns for col in ['ndvi_anterior', 'lluvia', 'temperatura']):
                        X = df[['ndvi_anterior', 'lluvia', 'temperatura']]
                        y = df['ndvi_actual']

                        model = LinearRegression()
                        model.fit(X, y)

                        df['ndvi_predicho'] = model.predict(X)

                        # Mostrar predicciones
                        st.subheader("🔮 Predicción de NDVI futuro")
                        st.dataframe(df[['zona', 'cultivo', 'ndvi_anterior', 'lluvia', 'temperatura', 'ndvi_predicho']])
                    else:
                        st.warning("⚠️ No se encontraron las columnas necesarias para predecir NDVI.")

                    # Mostrar valores
                    st.markdown(f"🌿 NDVI: **{ndvi_valor}**")
                    st.markdown(f"🌧️ {lluvia_valor} mm | 🌡️ {temp_valor}°C")

                    # Verificar umbrales
                    try:
                        riesgo = False
                        if ndvi_valor is not None and float(ndvi_valor) < 0.45:
                            riesgo = True
                        if lluvia_valor is not None and float(lluvia_valor) > 70:
                            riesgo = True
                        if temp_valor is not None and float(temp_valor) > 30:
                            riesgo = True

                        if riesgo:
                            st.error("🔴 Riesgo detectado:")
                        else:
                            st.success("🟢 No se detecta riesgo relevante.")

                        # Mostrar recomendación por cultivo
                        if cultivo in recomendaciones_por_cultivo:
                            st.info(f"💡 Recomendación para cultivo de {cultivo}:")
                            for nivel, recomendacion in recomendaciones_por_cultivo[cultivo].items():
                                st.markdown(f"- **{nivel}**: {recomendacion}")
                    except Exception as e:
                        st.error(f"Error en análisis: {e}")
                else:
                    st.warning("No hay datos para esta zona seleccionada.")


# =================== 🤖 Análisis automático de riesgo por zona y cultivo ===================
if "df_agricola" in st.session_state:
    df_agricola = st.session_state.df_agricola

    st.markdown("### 🤖 Análisis por zona y cultivo")

    zonas = df_agricola["zona"].unique()
    zona_sel = st.selectbox("Selecciona una zona:", zonas)

    # Guardar y recuperar la última selección de cultivo por zona
    if "cultivos_por_zona" not in st.session_state:
        st.session_state.cultivos_por_zona = {}

    cultivos_en_zona = df_agricola[df_agricola["zona"] == zona_sel]["cultivo"].unique()
    cultivo_default = st.session_state.cultivos_por_zona.get(zona_sel, cultivos_en_zona[0])
    cultivo_sel = st.selectbox("Selecciona el cultivo a analizar:", cultivos_en_zona, index=list(cultivos_en_zona).index(cultivo_default))
    st.session_state.cultivos_por_zona[zona_sel] = cultivo_sel

    if st.button("🔎 Ejecutar análisis automático"):
        datos = df_agricola[(df_agricola["zona"] == zona_sel) & (df_agricola["cultivo"] == cultivo_sel)]
        if not datos.empty:
            ndvi_valor = datos["ndvi_anterior"].values[-1]
            lluvia = datos["lluvia"].values[-1]
            temp = datos["temperatura"].values[-1]

            st.markdown(f"📍 Zona: {zona_sel} — 🌱 Cultivo: {cultivo_sel}")
            st.markdown(f"🌿 NDVI: {ndvi_valor} | 🌧️ {lluvia} mm | 🌡️ {temp}°C")

            riesgo = False
            if float(ndvi_valor) < 0.45 or float(lluvia) > 70 or float(temp) > 30:
                riesgo = True

            if riesgo:
                st.error("⚠️ Riesgo detectado en esta combinación zona + cultivo.")
            else:
                st.success("✅ Condiciones adecuadas.")

            if cultivo_sel in recomendaciones_por_cultivo:
                st.info(f"💡 Recomendación para cultivo {cultivo_sel}:")
                for nivel, mensaje in recomendaciones_por_cultivo[cultivo_sel].items():
                    st.markdown(f"- **{nivel}**: {mensaje}")
        else:
            st.warning("No hay datos suficientes para esta combinación.")