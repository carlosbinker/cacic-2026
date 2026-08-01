"""
Prompt de sistema utilizado para los cuatro modelos.

IMPORTANTE SOBRE FIDELIDAD -- leer antes de usar este módulo:

El paper (Sección 3.5, "Prompt utilizado") aclara explícitamente que el
prompt real usado en el estudio incluía DOS ejemplos few-shot, y que en el
texto publicado se reproduce "una versión condensada" con el primer
ejemplo únicamente, omitiendo el segundo "por espacio". El texto de ese
primer ejemplo SÍ está transcripto literalmente en el paper y se reproduce
tal cual abajo (SYSTEM_PROMPT_PAPER, un solo ejemplo).

El segundo ejemplo few-shot original no fue recuperable: no existe ningún
script guardado de la ejecución original, solo sus resultados (ver
data/resultados_experimento_detalle.csv). SYSTEM_PROMPT_RECONSTRUIDO abajo
agrega un segundo ejemplo razonable, cubriendo el caso de "ajuste sin
valor explícito" (la categoría que el paper identifica como más débil),
para que el pipeline sea utilizable de punta a punta. Está marcado como
adición propia, no como texto original del paper -- si vas a citar el
prompt exacto usado en el paper, usá SYSTEM_PROMPT_PAPER.
"""

SYSTEM_PROMPT_PAPER = (
    'Sos un módulo de interpretación de comandos de voz para un sistema '
    'domótico en español rioplatense. Convertí el comando del usuario en '
    'UN objeto JSON con estos campos: "intent" (encender/apagar/ajustar/'
    'consultar), "dispositivo" (luz, aire_acondicionado, calefaccion, '
    'persiana, cortina, tv, parlante, cerradura, ventilador, todos), '
    '"ubicacion" (living, cocina, dormitorio, bano, garage, patio, '
    'entrada, todas), "valor" (número si el comando lo da, si no null), '
    '"unidad" ("%" o "°C" si corresponde, si no null). Reglas: respondé '
    'SOLO el JSON, sin texto adicional; si el comando ajusta sin dar un '
    'número, "valor" y "unidad" deben ser null.\n\n'
    'Ejemplo: Comando: "Poné el aire de la cocina en 20 grados."\n'
    '{"intent": "ajustar", "dispositivo": "aire_acondicionado", '
    '"ubicacion": "cocina", "valor": 20, "unidad": "°C"}'
)

# Segundo ejemplo few-shot -- adición propia, NO transcripto del paper
# (ver docstring del módulo). Cubre el caso "ajuste relativo sin valor",
# que es exactamente la categoría con peor desempeño en los cuatro
# modelos (Tabla 3 del paper, 0%-16,7% de exactitud).
_SEGUNDO_EJEMPLO_RECONSTRUIDO = (
    '\n\nEjemplo 2: Comando: "Bajale un poco a la luz del living."\n'
    '{"intent": "ajustar", "dispositivo": "luz", "ubicacion": "living", '
    '"valor": null, "unidad": null}'
)

SYSTEM_PROMPT_RECONSTRUIDO = SYSTEM_PROMPT_PAPER + _SEGUNDO_EJEMPLO_RECONSTRUIDO


def construir_prompt_usuario(comando: str) -> str:
    """Formato del turno de usuario, según Sección 3.5: cada comando se
    envía como 'Comando: "..."' inmediatamente después del prompt de
    sistema, sin turnos previos de conversación."""
    return f'Comando: "{comando}"'
