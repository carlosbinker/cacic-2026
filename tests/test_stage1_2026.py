"""Tests de la consolidación de etapa 1 y de la selección del juez (F5/RF7)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import MODELOS_2026  # noqa: E402
from run_sweep_2026 import COLUMNAS_DETALLE  # noqa: E402
from stage1_2026 import (  # noqa: E402
    consolidar_detalle,
    elegir_juez,
    resumen_etapa1,
)


def _detalle(modelo: str, exactas: int, n: int = 32, latencia: float = 1.0) -> pd.DataFrame:
    """Detalle sintético: las primeras `exactas` filas aciertan, el resto no."""
    filas = []
    for i in range(n):
        ok = i < exactas
        filas.append({
            "modelo": modelo, "idx": i, "comando": f"c{i}", "gt": "{}",
            "pred_raw": "{}", "pred_json": "{}", "json_valido": True,
            "parse_note": "", "latencia_s": latencia,
            "match_intent": ok, "match_dispositivo": ok, "match_ubicacion": ok,
            "match_valor": ok, "match_unidad": ok, "match_exact": ok,
            "modo_prompting": "chat_template", "transformers_version": "4.57.0",
        })
    return pd.DataFrame(filas, columns=COLUMNAS_DETALLE)


def test_consolidar_ordena_por_roster_y_por_idx(tmp_path):
    d = tmp_path / "detalle"
    d.mkdir()
    # se escriben en orden inverso al roster a propósito
    for m in reversed(MODELOS_2026[:3]):
        _detalle(m.nombre, 10).to_csv(d / f"{m.nombre}.csv", index=False)
    df = consolidar_detalle(d, [m.nombre for m in MODELOS_2026[:3]])
    assert list(df.columns) == COLUMNAS_DETALLE
    assert df["modelo"].drop_duplicates().tolist() == [m.nombre for m in MODELOS_2026[:3]]
    assert df.groupby("modelo")["idx"].apply(list).map(lambda v: v == list(range(32))).all()


def test_consolidar_falla_si_falta_un_modelo(tmp_path):
    d = tmp_path / "detalle"
    d.mkdir()
    _detalle(MODELOS_2026[0].nombre, 5).to_csv(d / f"{MODELOS_2026[0].nombre}.csv", index=False)
    with pytest.raises(ValueError, match="faltan"):
        consolidar_detalle(d, [m.nombre for m in MODELOS_2026[:2]])


def test_resumen_etapa1_calcula_los_porcentajes(tmp_path):
    df = pd.concat([_detalle("A", 16, latencia=2.0), _detalle("B", 8, latencia=4.0)])
    filas = resumen_etapa1(df, {"A": 1.0, "B": 2.0})
    por_modelo = {f["modelo"]: f for f in filas}
    assert por_modelo["A"]["exact_match_pct"] == 50.0
    assert por_modelo["B"]["exact_match_pct"] == 25.0
    assert por_modelo["A"]["avg_latencia_s"] == 2.0
    assert por_modelo["A"]["n"] == 32
    assert por_modelo["A"]["json_valido_pct"] == 100.0


from stage1_2026 import calcular_filas_esperadas, nombres_disponibles


def test_calcula_384_filas_cuando_no_hubo_fallos():
    assert calcular_filas_esperadas([]) == 384


def test_descuenta_32_filas_por_cada_modelo_fallido():
    fallos = [
        {"modelo": "m1", "hf_repo_id": "x", "transformers_pin": "y",
         "codigo_salida": 1, "error_textual": "boom", "momento_iso": "2026-08-04T00:00:00"},
    ]
    assert calcular_filas_esperadas(fallos) == 352


def test_nombres_disponibles_excluye_a_los_modelos_fallidos():
    roster = [m.nombre for m in MODELOS_2026[:5]]
    fallos = [{"modelo": roster[2], "hf_repo_id": "x", "transformers_pin": "y",
               "codigo_salida": 1, "error_textual": "boom", "momento_iso": "t"}]
    disponibles = nombres_disponibles(roster, fallos)
    assert roster[2] not in disponibles
    assert len(disponibles) == len(roster) - 1


def test_nombres_disponibles_sin_fallos_devuelve_el_roster_completo():
    roster = [m.nombre for m in MODELOS_2026[:4]]
    assert nombres_disponibles(roster, []) == roster


def test_elige_al_de_mayor_exactitud():
    resumen = [
        {"modelo": "A", "params_b": 2.0, "exact_match_pct": 40.0},
        {"modelo": "B", "params_b": 0.5, "exact_match_pct": 55.0},
        {"modelo": "C", "params_b": 1.0, "exact_match_pct": 12.5},
    ]
    juez = elegir_juez(resumen)
    assert juez["modelo"] == "B"
    assert juez["empatados"] == ["B"]


def test_desempata_por_el_modelo_mas_grande():
    resumen = [
        {"modelo": "chico", "params_b": 0.35, "exact_match_pct": 60.0},
        {"modelo": "grande", "params_b": 1.7, "exact_match_pct": 60.0},
        {"modelo": "otro", "params_b": 1.2, "exact_match_pct": 59.9},
    ]
    juez = elegir_juez(resumen)
    assert juez["modelo"] == "grande"
    assert sorted(juez["empatados"]) == ["chico", "grande"]


def test_desempate_persistente_usa_el_orden_del_roster():
    """Misma exactitud y mismo tamaño: gana el que aparece antes en el roster."""
    a, b = MODELOS_2026[2].nombre, MODELOS_2026[7].nombre
    resumen = [
        {"modelo": b, "params_b": 1.0, "exact_match_pct": 50.0},
        {"modelo": a, "params_b": 1.0, "exact_match_pct": 50.0},
    ]
    assert elegir_juez(resumen)["modelo"] == a


def test_el_juez_trae_el_repo_y_el_criterio():
    nombre = MODELOS_2026[0].nombre
    juez = elegir_juez([{"modelo": nombre, "params_b": 0.23, "exact_match_pct": 10.0}])
    assert juez["hf_repo_id"] == MODELOS_2026[0].hf_repo_id
    assert "desempate" in juez["criterio"]
    assert set(juez) == {"modelo", "hf_repo_id", "params_b", "exact_match_pct",
                         "criterio", "empatados"}


def test_elegir_juez_es_deterministico_ante_el_orden_de_entrada():
    resumen = [
        {"modelo": MODELOS_2026[i].nombre, "params_b": MODELOS_2026[i].params_b,
         "exact_match_pct": 50.0}
        for i in range(4)
    ]
    assert elegir_juez(resumen)["modelo"] == elegir_juez(list(reversed(resumen)))["modelo"]


def test_elegir_juez_rechaza_resumen_vacio():
    with pytest.raises(ValueError, match="vacío"):
        elegir_juez([])
