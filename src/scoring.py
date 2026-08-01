"""
Lógica de parseo y scoring, separada de run_evaluation.py a propósito:
no depende de torch/transformers, así que se puede testear (y reutilizar
en metrics.py) sin necesidad de instalar ni descargar nada pesado.
"""

import json
import re

import pandas as pd

from schema import CAMPOS_ESQUEMA


def fila_a_ground_truth(fila: pd.Series) -> dict:
    """Arma el dict de ground truth con los 5 campos del esquema, a partir
    de una fila del CSV del dataset (valor/unidad vacíos -> None, como
    especifica la Sección 3.2 del paper)."""
    return {
        "intent": fila["intent"],
        "dispositivo": fila["dispositivo"],
        "ubicacion": fila["ubicacion"],
        "valor": None if pd.isna(fila["valor"]) else fila["valor"],
        "unidad": None if pd.isna(fila["unidad"]) else fila["unidad"],
    }


def extraer_json(texto_generado: str) -> tuple[dict | None, bool, str]:
    """Intenta parsear la respuesta del modelo como el objeto JSON pedido.

    Devuelve (dict_parseado_o_None, json_valido, nota). Se intenta primero
    un parseo directo (lo esperado si el modelo respeta la regla "SOLO el
    JSON, sin texto adicional"), y si falla se intenta limpiar envoltorios
    comunes (fences de markdown, texto antes/después del objeto) antes de
    declarar inválido.
    """
    texto = texto_generado.strip()

    try:
        return json.loads(texto), True, ""
    except json.JSONDecodeError:
        pass

    match_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match_fence:
        try:
            return json.loads(match_fence.group(1)), True, "extraido_de_fence_markdown"
        except json.JSONDecodeError:
            pass

    match_obj = re.search(r"\{.*\}", texto, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group(0)), True, "extraido_de_texto_con_ruido"
        except json.JSONDecodeError:
            pass

    return None, False, "no_parseable_como_json"


def comparar_campos(pred: dict | None, gt: dict) -> dict:
    """Calcula match por campo y coincidencia exacta, replicando las
    métricas (2) y (3) de la Sección 3.4 del paper."""
    if pred is None:
        return {
            "match_intent": False,
            "match_dispositivo": False,
            "match_ubicacion": False,
            "match_valor": False,
            "match_unidad": False,
            "match_exact": False,
        }

    matches = {
        f"match_{campo}": pred.get(campo) == gt[campo]
        for campo in CAMPOS_ESQUEMA
    }
    matches["match_exact"] = all(matches.values())
    return matches
