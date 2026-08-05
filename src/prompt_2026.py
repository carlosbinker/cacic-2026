"""
Prompt de sistema 2026 y construcción de entrada por capacidad del tokenizer.

Extiende --- sin modificarlo --- el prompt publicado en la Sección 3.5 del
paper, que se importa de src/prompt.py (archivo congelado). Los dos agregados
son: (a) la cláusula taxativa sobre campos categóricos cerrados, que es un
cambio deliberado de protocolo respecto del estudio original, y (b) el segundo
ejemplo few-shot que el paper declara haber usado pero no transcribe.

Como el roster 2026 incluye modelos base que no exponen chat_template, la
construcción de entrada despacha por capacidad y no por identificador de
modelo.
"""

from typing import Literal

from prompt import SYSTEM_PROMPT_PAPER, construir_prompt_usuario

ModoPrompting = Literal["chat_template", "raw_completion"]

CLAUSULA_TAXATIVA = (
    'IMPORTANTE: los campos "intent", "dispositivo", "ubicacion" y "unidad" '
    "son campos CATEGORICOS CERRADOS. Los valores listados entre parentesis "
    "mas arriba son, textualmente, los unicos outputs aceptados para esos "
    "campos: copialos exactamente como aparecen, sin tildes, sin mayusculas, "
    "sin plural y sin reformular. Usar un sinonimo, una traduccion o "
    "cualquier variante que no este literalmente en esa lista (por ejemplo "
    '"televisor" o "tele" en lugar de "tv") es una violacion del formato y '
    "la respuesta se considera incorrecta. Si ningun valor de la lista "
    "aplica, usa null."
)

_SEGUNDO_EJEMPLO = (
    'Ejemplo 2: Comando: "Bajale un poco a la luz del living."\n'
    '{"intent": "ajustar", "dispositivo": "luz", "ubicacion": "living", '
    '"valor": null, "unidad": null}'
)

SYSTEM_PROMPT_2026 = (
    SYSTEM_PROMPT_PAPER + "\n\n" + CLAUSULA_TAXATIVA + "\n\n" + _SEGUNDO_EJEMPLO
)

RAW_TEMPLATE = '{system}\n\nComando: "{comando}"\nJSON:'


def detectar_modo(tokenizer) -> ModoPrompting:
    """Decide el modo por capacidad, nunca por identificador de modelo.

    Los modelos base del roster no traen chat_template; hardcodear sus IDs
    haría que el harness se rompa en silencio si mañana el proveedor publica
    una variante instruct con el mismo nombre."""
    plantilla = getattr(tokenizer, "chat_template", None)
    return "raw_completion" if not plantilla else "chat_template"


def construir_entrada_con_prompt(tokenizer, comando: str,
                                 system_prompt: str) -> tuple[dict, ModoPrompting]:
    """Igual que `construir_entrada`, pero con el prompt de sistema como
    parámetro en vez de hardcodeado a `SYSTEM_PROMPT_2026`.

    Factorización mínima (Delta 04 / corrida de control): existe para que
    `src/run_control_prompt_original.py` pueda reusar exactamente la misma
    lógica de despacho por capacidad (chat_template vs. raw_completion) y
    sustituir únicamente el texto del prompt, sin copiar esta función. La
    firma pública de `construir_entrada` (F2, congelada) no cambia.
    """
    if not comando or not comando.strip():
        raise ValueError(f"Se recibió un comando vacío: {comando!r}")

    modo = detectar_modo(tokenizer)
    if modo == "chat_template":
        mensajes = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": construir_prompt_usuario(comando)},
        ]
        entrada = tokenizer.apply_chat_template(
            mensajes,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        texto = RAW_TEMPLATE.format(system=system_prompt, comando=comando)
        entrada = tokenizer(texto, return_tensors="pt")
    return entrada, modo


def construir_entrada(tokenizer, comando: str) -> tuple[dict, ModoPrompting]:
    """Devuelve (entrada, modo), con entrada apta para modelo.generate(**entrada)."""
    return construir_entrada_con_prompt(tokenizer, comando, SYSTEM_PROMPT_2026)
