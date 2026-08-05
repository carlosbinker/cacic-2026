"""
Prompt de sistema 2026 y construcción de entrada por capacidad del tokenizer.

Extiende --- sin modificarlo --- el prompt publicado en la Sección 3.5 del
paper, que se importa de src/prompt.py (archivo congelado). Los dos agregados
son: (a) la cláusula taxativa sobre campos categóricos cerrados, que es un
cambio deliberado de protocolo respecto del estudio original, y (b) el segundo
ejemplo few-shot que el paper declara haber usado pero no transcribe.

Revertido (2026-08-05, instrucción del usuario): existió una ruta de
prompting alternativa por *raw completion* para tokenizers sin
`chat_template`, pensada para modelos base. Los 12 modelos del roster 2026
son variantes instruct/chat y exponen `chat_template`; ningún modelo la
ejerció nunca, y su única cobertura era un test sintético con un tokenizer
falso. Se eliminó esa ruta: `construir_entrada` aplica la plantilla de chat
directamente y falla explícitamente -- no con un `AttributeError`/`TypeError`
opaco -- si el tokenizer no la expone. Si en el futuro hace falta evaluar un
modelo base de verdad, hay que reincorporar una ruta de raw completion
explícita, no reintroducir un fallback silencioso.
"""

from prompt import SYSTEM_PROMPT_PAPER, construir_prompt_usuario

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

def construir_entrada_con_prompt(tokenizer, comando: str, system_prompt: str) -> dict:
    """Igual que `construir_entrada`, pero con el prompt de sistema como
    parámetro en vez de hardcodeado a `SYSTEM_PROMPT_2026`.

    Factorización mínima (Delta 04 / corrida de control): existe para que
    `src/run_control_prompt_original.py` pueda reusar exactamente esta misma
    lógica y sustituir únicamente el texto del prompt, sin copiar esta
    función. La firma pública de `construir_entrada` (F2, congelada) no
    cambia en lo que le importa a sus llamadores: recibe tokenizer y comando,
    devuelve algo apto para `modelo.generate(**entrada)`.
    """
    if not comando or not comando.strip():
        raise ValueError(f"Se recibió un comando vacío: {comando!r}")

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "El tokenizer no expone chat_template: el harness 2026 solo admite "
            "modelos instruct/chat con plantilla de chat (la ruta de raw "
            "completion para modelos base se eliminó el 2026-08-05 porque "
            "ningún modelo del roster la ejercía). Si hace falta evaluar un "
            "modelo base de verdad, hay que reincorporar una ruta de raw "
            "completion explícita en vez de forzarlo por esta función."
        )

    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": construir_prompt_usuario(comando)},
    ]
    return tokenizer.apply_chat_template(
        mensajes,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def construir_entrada(tokenizer, comando: str) -> dict:
    """Entrada apta para `modelo.generate(**entrada)`, con el prompt 2026."""
    return construir_entrada_con_prompt(tokenizer, comando, SYSTEM_PROMPT_2026)
