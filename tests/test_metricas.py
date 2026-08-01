"""
Tests unitarios para la lógica de scoring (src/metrics.py) y de parseo de
JSON (src/run_evaluation.py). Corré con: pytest
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import calcular_resumen  # noqa: E402
from scoring import comparar_campos, extraer_json  # noqa: E402


def test_extraer_json_directo():
    pred, valido, nota = extraer_json(
        '{"intent": "encender", "dispositivo": "luz", "ubicacion": "living", "valor": null, "unidad": null}'
    )
    assert valido is True
    assert nota == ""
    assert pred["intent"] == "encender"


def test_extraer_json_con_fence_markdown():
    texto = '```json\n{"intent": "apagar", "dispositivo": "tv", "ubicacion": "living", "valor": null, "unidad": null}\n```'
    pred, valido, nota = extraer_json(texto)
    assert valido is True
    assert nota == "extraido_de_fence_markdown"
    assert pred["dispositivo"] == "tv"


def test_extraer_json_invalido():
    pred, valido, nota = extraer_json("esto no es un JSON en absoluto")
    assert valido is False
    assert pred is None


def test_comparar_campos_coincidencia_exacta():
    gt = {"intent": "ajustar", "dispositivo": "aire_acondicionado",
          "ubicacion": "cocina", "valor": 20, "unidad": "°C"}
    pred = dict(gt)
    matches = comparar_campos(pred, gt)
    assert matches["match_exact"] is True
    assert all(matches.values())


def test_comparar_campos_alucinacion_de_valor():
    """El caso central del paper: el modelo inventa un valor cuando el
    ground truth dice que debería ser null."""
    gt = {"intent": "ajustar", "dispositivo": "calefaccion",
          "ubicacion": "dormitorio", "valor": None, "unidad": None}
    pred = {"intent": "ajustar", "dispositivo": "calefaccion",
            "ubicacion": "dormitorio", "valor": 2, "unidad": "°C"}
    matches = comparar_campos(pred, gt)
    assert matches["match_intent"] is True
    assert matches["match_valor"] is False
    assert matches["match_unidad"] is False
    assert matches["match_exact"] is False


def test_comparar_campos_prediccion_nula():
    gt = {"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
          "valor": None, "unidad": None}
    matches = comparar_campos(None, gt)
    assert matches["match_exact"] is False
    assert matches["match_intent"] is False


def test_calcular_resumen_coincide_con_resultados_reales():
    """Validación de regresión: la agregación debe reproducir exactamente
    el resumen real publicado, dado el CSV de detalle real."""
    raiz = Path(__file__).resolve().parent.parent
    df_detalle = pd.read_csv(raiz / "data" / "resultados_experimento_detalle.csv")
    resumen = {fila["modelo"]: fila for fila in calcular_resumen(df_detalle)}

    assert resumen["SmolLM2-1.7B-Instruct"]["exact_match_pct"] == pytest.approx(59.4)
    assert resumen["Qwen2.5-1.5B-Instruct"]["exact_match_pct"] == pytest.approx(50.0)
    assert resumen["Qwen2.5-0.5B-Instruct"]["exact_match_pct"] == pytest.approx(43.8)
    assert resumen["SmolLM2-360M-Instruct"]["exact_match_pct"] == pytest.approx(18.8)
    for fila in resumen.values():
        assert fila["json_valido_pct"] == pytest.approx(100.0)
