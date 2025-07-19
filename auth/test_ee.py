import ee
import json

# Ruta al archivo JSON (ajusta según donde lo pongas)
CREDENTIALS_PATH = 'auth/credentials.json'
SERVICE_ACCOUNT = 'earthengine-service@agtech-ecuador.iam.gserviceaccount.com'

# Cargar las credenciales del archivo JSON
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, CREDENTIALS_PATH)

# Inicializar Earth Engine
try:
    ee.Initialize(credentials)
    print("✅ Conexión exitosa con Google Earth Engine.")

    # Probar lectura de una imagen Sentinel
    image = ee.Image("COPERNICUS/S2/20190830T142139_20190830T142141_T18MYJ")
    info = image.getInfo()
    print("🛰 Imagen recuperada correctamente:")
    print(info['id'])

except Exception as e:
    print("❌ Error al conectar con Earth Engine:")
    print(e)
