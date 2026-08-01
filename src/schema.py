"""
Esquema del JSON de salida y vocabulario cerrado del dominio.

Reproduce exactamente el vocabulario especificado en el prompt de sistema
descripto en la Sección 3.5 del paper ("Prompt utilizado"). Los valores
literales se mantienen deliberadamente sin tildes ni la letra ñ (p. ej.
"bano", "calefaccion", "ubicacion") para evitar inconsistencias de
tokenización entre los cuatro modelos evaluados -- la misma decisión que
documenta el paper en su nota al pie de la Sección 3.5.
"""

INTENTS = ["encender", "apagar", "ajustar", "consultar"]

DISPOSITIVOS = [
    "luz",
    "aire_acondicionado",
    "calefaccion",
    "persiana",
    "cortina",
    "tv",
    "parlante",
    "cerradura",
    "ventilador",
    "todos",
]

UBICACIONES = [
    "living",
    "cocina",
    "dormitorio",
    "bano",
    "garage",
    "patio",
    "entrada",
    "todas",
]

UNIDADES = ["%", "°C"]

# Los cinco campos de la estructura de referencia (ground truth), en el
# mismo orden que la Sección 3.2 del paper.
CAMPOS_ESQUEMA = ["intent", "dispositivo", "ubicacion", "valor", "unidad"]
