"""Vocabularios cerrados de la etapa 2 del experimento 2026 (F4 del índice).

Define las 7 etiquetas de error y las 6 categorías lingüísticas cerradas que
usa el juez LLM, junto con los parseadores que validan su salida contra esos
vocabularios. Puro dato + validación: sin dependencias de `torch`/`transformers`.
"""

ETIQUETAS_ERROR: list[str] = [
    "confusion_intencion",
    "confusion_dispositivo",
    "confusion_ubicacion",
    "alucinacion_valor_unidad",
    "valor_numerico_incorrecto",
    "sin_error_semantico",
    "uso_de_sinonimos",
]
ETIQUETAS_EQUIVALENTES: frozenset[str] = frozenset({"sin_error_semantico", "uso_de_sinonimos"})

CATEGORIAS_LINGUISTICAS: list[str] = [
    "encendido_apagado_simple",
    "ajuste_con_valor_numerico",
    "ajuste_relativo_cualitativo",
    "multiples_dispositivos",
    "consulta_de_estado",
    "negacion_contexto",
]

ETIQUETAS_ERROR_DISPLAY: dict[str, str] = {
    "confusion_intencion": "Confusión de intención",
    "confusion_dispositivo": "Confusión de dispositivo",
    "confusion_ubicacion": "Confusión de ubicación",
    "alucinacion_valor_unidad": "Alucinación de valor/unidad",
    "valor_numerico_incorrecto": "Valor numérico incorrecto",
    "sin_error_semantico": "Sin error semántico",
    "uso_de_sinonimos": "Uso de sinónimos",
}
CATEGORIAS_DISPLAY: dict[str, str] = {
    "encendido_apagado_simple": "Encendido / apagado simple",
    "ajuste_con_valor_numerico": "Ajuste con valor numérico",
    "ajuste_relativo_cualitativo": "Ajuste relativo / cualitativo",
    "multiples_dispositivos": "Múltiples dispositivos",
    "consulta_de_estado": "Consulta de estado",
    "negacion_contexto": "Negación / contexto conversacional",
}

SEP_ETIQUETAS: str = ";"


def parsear_etiquetas(texto: str) -> list[str]:
    """Parsea la salida del juez contra el vocabulario cerrado ETIQUETAS_ERROR.

    Normaliza a minúsculas y descarta duplicados preservando el orden de
    aparición. Es deliberadamente estricto: cualquier etiqueta fuera del
    vocabulario es un error, no un valor a ignorar en silencio."""
    crudas = [p.strip().lower() for p in texto.split(SEP_ETIQUETAS)]
    crudas = [p for p in crudas if p]
    if not crudas:
        raise ValueError(f"Se esperaba al menos una etiqueta, se recibió: {texto!r}")

    vistas: list[str] = []
    for etiqueta in crudas:
        if etiqueta not in ETIQUETAS_ERROR:
            raise ValueError(
                f"Etiqueta fuera del vocabulario cerrado: {etiqueta!r}. "
                f"Válidas: {', '.join(ETIQUETAS_ERROR)}"
            )
        if etiqueta not in vistas:
            vistas.append(etiqueta)
    return vistas


def parsear_categoria(texto: str) -> str:
    """Parsea la salida del juez contra CATEGORIAS_LINGUISTICAS: exactamente una."""
    crudas = [p.strip().lower() for p in texto.split(SEP_ETIQUETAS) if p.strip()]
    if len(crudas) != 1:
        raise ValueError(
            f"Se esperaba exactamente una categoría, se recibieron {len(crudas)}: {texto!r}"
        )
    categoria = crudas[0]
    if categoria not in CATEGORIAS_LINGUISTICAS:
        raise ValueError(
            f"Categoría fuera del vocabulario cerrado: {categoria!r}. "
            f"Válidas: {', '.join(CATEGORIAS_LINGUISTICAS)}"
        )
    return categoria
