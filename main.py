# Crear el nuevo contenido mejorado de main.py
nuevo_main_code = '''
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 1. Función de evaluación
# --------------------------
def evaluar_riesgo(ndvi_actual, ndvi_anterior, lluvia, temperatura):
    alerta = False
    razones = []

    if ndvi_anterior != 0 and (ndvi_actual - ndvi_anterior) / ndvi_anterior <= -0.2:
        alerta = True
        razones.append("Caída de NDVI ≥ 20%")

    if temperatura > 28 and lluvia > 80:
        alerta = True
        razones.append("Condiciones favorables a hongos/plagas")

    return alerta, ", ".join(razones)

# --------------------------
# 2. Leer y validar CSV
# --------------------------
try:
    df = pd.read_csv("datos_agricolas.csv")
except FileNotFoundError:
    print("❌ Archivo CSV no encontrado.")
    exit()
except Exception as e:
    print(f"❌ Error al leer el CSV: {e}")
    exit()

columnas_requeridas = ["ndvi_actual", "ndvi_anterior", "lluvia", "temperatura", "zona"]
for col in columnas_requeridas:
    if col not in df.columns:
        print(f"❌ Falta la columna requerida: {col}")
        exit()

# --------------------------
# 3. Evaluación de riesgo
# --------------------------
df["alerta"], df["razones"] = zip(*df.apply(
    lambda row: evaluar_riesgo(
        row["ndvi_actual"],
        row["ndvi_anterior"],
        row["lluvia"],
        row["temperatura"]
    ), axis=1
))

# Imprimir resultados
for index, row in df.iterrows():
    print(f"\\n📍 Zona: {row['zona']}")
    if row["alerta"]:
        print("⚠️ Riesgo detectado:")
        for razon in row["razones"].split(", "):
            print(" -", razon)
    else:
        print("✅ Sin riesgo.")

# Exportar resultados
df.to_csv("riesgos_detectados.csv", index=False)
print("\\n📁 Resultados exportados a 'riesgos_detectados.csv'")

# --------------------------
# 4. Graficar NDVI por zona
# --------------------------
if "fecha" in df.columns:
    zonas = df["zona"].unique()
    for zona in zonas:
        sub_df = df[df["zona"] == zona]
        plt.plot(sub_df["fecha"], sub_df["ndvi_actual"], marker='o', label=zona)

    plt.axhline(0.6, color='r', linestyle='--', label="Umbral de alerta")
    plt.title("Evolución NDVI por zona")
    plt.ylabel("NDVI")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
else:
    print("ℹ️ No se encontró la columna 'fecha' para graficar NDVI.")
'''

# Guardar el nuevo archivo main.py sobrescribiendo el anterior
with open(main_file_path, 'w', encoding='utf-8') as file:
    file.write(nuevo_main_code)

nuevo_main_code[:3000]  # Mostrar parte del nuevo archivo generado
