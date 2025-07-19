import ee

# Autenticación manual con código
ee.Authenticate(auth_mode='notebook')
ee.Initialize()

print("✅ Earth Engine autenticado correctamente.")