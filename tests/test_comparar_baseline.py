"""Tests de `src/comparar_baseline.py` (Delta 05, subtask 19, Tarea 4).

Compara, para los 4 modelos de `roster_baseline_original()`, la cifra
PUBLICADA (Tabla 2 original -- ahora referencia HISTÓRICA, no reproducida:
ver `tests/test_baseline_original.py` y el módulo bajo prueba para la
evidencia) contra la RE-MEDIDA bajo el harness/prompt 2026, y determina por
CÓDIGO cuál es el mejor original re-medido, para que la comparación central
del paper (mejor arquitectura 2026 vs mejor original re-medido) no se
afirme a mano.

Sin inferencia real y sin escribir bajo `data/2026/`: los tests con datos
sintéticos usan `tmp_path`/`monkeypatch`; los que usan datos reales solo
LEEN los CSV ya commiteados (`data/2026/detalle/`, de solo lectura) y el F0
(`data/resultados_experimento_resumen.json`).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import por_nombre, roster_baseline_original, slug  # noqa: E402

import comparar_baseline as cb  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def test_cargar_publicado_usa_el_f0_real_de_los_cuatro_modelos():
    publicado = cb.cargar_publicado()
    assert publicado["Qwen2.5-0.5B-Instruct"]["exact_match_pct"] == 43.8
    assert publicado["Qwen2.5-1.5B-Instruct"]["exact_match_pct"] == 50.0
    assert publicado["SmolLM2-1.7B-Instruct"]["exact_match_pct"] == 59.4
    assert publicado["SmolLM2-360M-Instruct"]["exact_match_pct"] == 18.8


def test_ruta_remedida_usa_detalle_para_activos_y_baseline_para_el_resto():
    activo = por_nombre("Qwen2.5-1.5B-Instruct")
    inactivo = por_nombre("Qwen2.5-0.5B-Instruct")
    assert cb.ruta_remedida(activo).parent.name == "detalle"
    assert cb.ruta_remedida(inactivo).parent.name == "baseline_original"


def test_comparar_modelo_con_datos_reales_de_un_ancla_ya_corrida():
    """SmolLM2-360M-Instruct ya tiene su CSV real en data/2026/detalle/."""
    publicado = cb.cargar_publicado()
    resultado = cb.comparar_modelo("SmolLM2-360M-Instruct", publicado)
    assert resultado["remedido"] is not None
    assert resultado["remedido"]["n"] == 32
    assert resultado["publicado"]["exact_match_pct"] == 18.8
    nota = resultado["publicado"]["nota"].lower()
    assert "referencia hist" in nota and "no reproduc" in nota


def test_comparar_modelo_da_none_si_el_csv_todavia_no_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "DIR_BASELINE", tmp_path / "baseline_original")
    (tmp_path / "baseline_original").mkdir()
    publicado = cb.cargar_publicado()
    resultado = cb.comparar_modelo("Qwen2.5-0.5B-Instruct", publicado)
    assert resultado["remedido"] is None
    assert resultado["delta_exact_match_pp"] is None


def test_comparar_modelo_rechaza_un_modelo_fuera_del_baseline_original():
    with pytest.raises(ValueError):
        cb.comparar_modelo("LFM2.5-230M", cb.cargar_publicado())


def test_comparar_todo_incluye_los_cuatro_modelos_en_orden_de_registro():
    comparacion = cb.comparar_todo()
    nombres = [f["modelo"] for f in comparacion["modelos"]]
    assert set(nombres) == {m.nombre for m in roster_baseline_original()}
    assert len(nombres) == 4


def test_mejor_remedido_se_calcula_por_codigo_no_se_hardcodea(tmp_path, monkeypatch):
    """Con datos sinteticos controlados, el ganador tiene que ser el de mayor
    exact_match_pct remedido, sea cual sea el nombre."""
    monkeypatch.setattr(cb, "DIR_DETALLE", tmp_path / "detalle")
    monkeypatch.setattr(cb, "DIR_BASELINE", tmp_path / "baseline_original")
    (tmp_path / "detalle").mkdir()
    (tmp_path / "baseline_original").mkdir()

    def _df(n_exactos, n=32):
        return pd.DataFrame([
            {"match_exact": i < n_exactos, "json_valido": True, "latencia_s": 1.0}
            for i in range(n)
        ])

    for nombre, n_exactos, activo in [
        ("SmolLM2-360M-Instruct", 3, True),
        ("SmolLM2-1.7B-Instruct", 30, True),
        ("Qwen2.5-1.5B-Instruct", 10, True),
        ("Qwen2.5-0.5B-Instruct", 5, False),
    ]:
        directorio = (tmp_path / "detalle") if activo else (tmp_path / "baseline_original")
        _df(n_exactos).to_csv(directorio / f"{slug(nombre)}.csv", index=False)

    comparacion = cb.comparar_todo()
    assert comparacion["mejor_remedido"]["modelo"] == "SmolLM2-1.7B-Instruct"
    assert comparacion["mejor_remedido"]["exact_match_pct"] == pytest.approx(30 / 32 * 100, abs=0.1)


def test_comparacion_central_lee_granite_4_0_1b_de_datos_reales():
    """El contraste central del paper (mejor arquitectura 2026 vs mejor original
    remedido) se computa por script, nunca se afirma a mano."""
    comparacion = cb.comparar_todo()
    central = comparacion["comparacion_central"]
    assert central["mejor_arquitectura_2026"]["modelo"] == "granite-4.0-1b"
    assert central["mejor_arquitectura_2026"]["exact_match_pct"] == pytest.approx(90.6, abs=0.1)
    assert central["mejor_original_remedido"]["modelo"] == comparacion["mejor_remedido"]["modelo"]
    assert central["delta_pp"] == round(
        central["mejor_arquitectura_2026"]["exact_match_pct"]
        - central["mejor_original_remedido"]["exact_match_pct"], 1
    )


def test_publicado_esta_marcado_como_referencia_historica_no_reproducida():
    texto = cb.NOTA_PUBLICADO.lower()
    assert "historica" in texto or "histórica" in texto
    assert "no reproduc" in texto


def test_escribir_comparacion_escribe_json_indentado(tmp_path):
    comparacion = {"modelos": [], "nota": "x"}
    salida = tmp_path / "out.json"
    cb.escribir_comparacion(comparacion, salida)
    assert json.loads(salida.read_text(encoding="utf-8")) == comparacion


def test_imprimir_tabla_no_crashea_con_bandas_faltantes(capsys):
    comparacion = {
        "modelos": [{
            "modelo": "X",
            "publicado": {"exact_match_pct": 10.0, "json_valido_pct": 100.0, "nota": cb.NOTA_PUBLICADO},
            "remedido": None,
            "delta_exact_match_pp": None,
            "delta_json_valido_pp": None,
        }],
        "mejor_remedido": None,
        "comparacion_central": None,
        "nota_publicado": cb.NOTA_PUBLICADO,
    }
    cb.imprimir_tabla(comparacion)
    assert "referencia" in capsys.readouterr().out.lower()


def test_main_escribe_y_no_toca_el_repo(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cb, "SALIDA_PATH", tmp_path / "comparacion_baseline.json")
    codigo = cb.main()
    assert codigo == 0
    assert (tmp_path / "comparacion_baseline.json").exists()
    assert "modelo" in capsys.readouterr().out.lower()


def test_el_modulo_de_comparacion_baseline_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "comparar_baseline.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea
