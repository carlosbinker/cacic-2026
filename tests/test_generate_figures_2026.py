"""Tests de las figuras 2026: preparación de datos y generación de archivos."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from generate_figures_2026 import (  # noqa: E402
    CAMPOS_FIG2,
    generar_figuras,
    ordenar_para_grafico,
    posiciones_con_separacion,
)
from models_2026 import roster_activo  # noqa: E402


def _resumen():
    filas = []
    for i, m in enumerate(roster_activo()):
        filas.append({
            "modelo": m.nombre, "hf_repo_id": m.hf_repo_id, "params_b": m.params_b,
            "tier": m.tier, "modo_prompting": "chat_template",
            "transformers_pin": m.transformers_pin, "n": 32,
            "json_valido_pct": 100.0,
            "exact_match_pct": float(10 + i * 4),
            "exact_match_laxo_pct": float(15 + i * 4),
            "avg_latencia_s": float(5 + i * 3),
            "acc_intent_pct": 80.0, "acc_dispositivo_pct": 70.0,
            "acc_ubicacion_pct": 75.0, "acc_valor_pct": 85.0, "acc_unidad_pct": 90.0,
        })
    return filas


def test_ordena_sub1b_primero_y_por_params_dentro_del_tier():
    ordenado = ordenar_para_grafico(_resumen())
    tiers = [f["tier"] for f in ordenado]
    assert tiers == ["sub-1B"] * 6 + ["1-2B"] * 6
    sub = [f["params_b"] for f in ordenado[:6]]
    grandes = [f["params_b"] for f in ordenado[6:]]
    assert sub == sorted(sub) and grandes == sorted(grandes)


def test_ordenar_no_pierde_ni_duplica_modelos():
    ordenado = ordenar_para_grafico(_resumen())
    assert len(ordenado) == 12
    assert {f["modelo"] for f in ordenado} == {m.nombre for m in roster_activo()}


def test_hay_un_hueco_entre_los_dos_tiers():
    tiers = ["sub-1B"] * 6 + ["1-2B"] * 6
    pos = posiciones_con_separacion(tiers)
    assert len(pos) == 12
    assert all(b > a for a, b in zip(pos, pos[1:]))          # monótono
    dentro = pos[1] - pos[0]
    entre = pos[6] - pos[5]
    assert entre > dentro, (entre, dentro)                   # hueco entre tiers


def test_posiciones_con_un_solo_tier_no_deja_hueco():
    pos = posiciones_con_separacion(["sub-1B"] * 3)
    assert [round(b - a, 6) for a, b in zip(pos, pos[1:])] == [1.0, 1.0]


def test_campos_fig2_son_los_cinco_del_esquema():
    assert CAMPOS_FIG2 == [
        ("acc_intent_pct", "intención"),
        ("acc_dispositivo_pct", "dispositivo"),
        ("acc_ubicacion_pct", "ubicación"),
        ("acc_valor_pct", "valor"),
        ("acc_unidad_pct", "unidad"),
    ]


def test_generar_figuras_escribe_los_dos_png(tmp_path):
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(_resumen(), ensure_ascii=False), encoding="utf-8")
    salidas = generar_figuras(resumen, tmp_path / "figs", sufijo="")
    assert len(salidas) == 2
    for ruta in salidas:
        assert ruta.exists() and ruta.stat().st_size > 5000
    assert salidas[0].name == "fig1_exactitud_latencia_2026.png"
    assert salidas[1].name == "fig2_exactitud_por_campo_2026.png"


def test_el_sufijo_no_pisa_las_figuras_oficiales(tmp_path):
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(_resumen(), ensure_ascii=False), encoding="utf-8")
    salidas = generar_figuras(resumen, tmp_path / "figs", sufijo="_reproducido")
    assert salidas[0].name == "fig1_exactitud_latencia_2026_reproducido.png"


def test_generar_figuras_falla_si_falta_la_metrica_laxa(tmp_path):
    datos = _resumen()
    for f in datos:
        del f["exact_match_laxo_pct"]
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact_match_laxo_pct"):
        generar_figuras(resumen, tmp_path / "figs", sufijo="")
