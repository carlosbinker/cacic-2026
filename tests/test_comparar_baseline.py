"""Tests de `src/comparar_baseline.py` (Delta 06).

`paper_cacic_LNCS_word.docx` es un BORRADOR del paper en escritura, no
trabajo publicado: sus cifras son valores de borrador SUPERADOS, no una
base de comparación, y este módulo NO es una auditoría de reproducibilidad
(esa framing del Delta 04/05 queda retirada). Los 4 modelos de
`roster_baseline_original()` son el brazo de generación ANTERIOR de este
mismo experimento, medido bajo el mismo harness/máquina/decodificación que
los 12 del roster activo, primero bajo el prompt ORIGINAL del borrador y
después bajo el prompt 2026. Este módulo arma la matriz 4×2 y calcula por
CÓDIGO el titular del paper (mejor generación anterior bajo el prompt 2026
vs `granite-4.0-1b`), con el prompt SIEMPRE fijo.

Sin inferencia real y sin escribir bajo `data/2026/`: los tests con datos
sintéticos usan `tmp_path`/`monkeypatch`; los que usan datos reales solo
LEEN los CSV ya commiteados (`data/2026/detalle/`, `data/2026/baseline_original/`,
`data/2026/control_prompt_original/`, todos de solo lectura).
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


def _df(n_exactos: int, n_json_validos: int, n: int = 32) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_exact": i < n_exactos,
            "json_valido": i < n_json_validos,
            "latencia_s": 1.0 + i,
        }
        for i in range(n)
    ])


# --------------------------------------------------------------------------
# Métricas y rutas de celda
# --------------------------------------------------------------------------

def test_metricas_de_detalle_calcula_exact_match_y_json_valido():
    m = cb.metricas_de_detalle(_df(n_exactos=8, n_json_validos=16, n=32))
    assert m["n"] == 32
    assert m["exact_match_pct"] == 25.0
    assert m["json_valido_pct"] == 50.0
    # el titular nunca usa latencia de estas bandas: no forma parte de la celda
    assert "avg_latencia_s" not in m


def test_ruta_celda_prompt_original_siempre_apunta_a_control_prompt_original():
    activo = por_nombre("Qwen2.5-1.5B-Instruct")
    inactivo = por_nombre("Qwen2.5-0.5B-Instruct")
    assert cb.ruta_celda(activo, "original").parent.name == "control_prompt_original"
    assert cb.ruta_celda(inactivo, "original").parent.name == "control_prompt_original"


def test_ruta_celda_prompt_2026_usa_detalle_para_activos_y_baseline_para_el_resto():
    activo = por_nombre("Qwen2.5-1.5B-Instruct")
    inactivo = por_nombre("Qwen2.5-0.5B-Instruct")
    assert cb.ruta_celda(activo, "2026").parent.name == "detalle"
    assert cb.ruta_celda(inactivo, "2026").parent.name == "baseline_original"


def test_cargar_celda_devuelve_none_si_el_csv_no_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "DIR_CONTROL", tmp_path)
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    assert cb.cargar_celda(modelo, "original") is None


# --------------------------------------------------------------------------
# Fila por modelo / matriz completa
# --------------------------------------------------------------------------

def test_fila_modelo_con_datos_reales_de_un_ancla_ya_corrida():
    """SmolLM2-360M-Instruct ya tiene sus dos celdas reales."""
    fila = cb.fila_modelo(por_nombre("SmolLM2-360M-Instruct"))
    assert fila["prompt_original"] is not None
    assert fila["prompt_2026"] is not None
    assert fila["prompt_original"]["exact_match_pct"] == 0.0
    assert fila["prompt_2026"]["exact_match_pct"] == 9.4
    assert fila["efecto_prompt_exact_match_pp"] == round(9.4 - 0.0, 1)


def test_fila_modelo_reporta_pendiente_la_celda_que_todavia_no_existe():
    """Qwen2.5-0.5B-Instruct todavía no tiene su CSV bajo el prompt original
    (lo agrega el orquestador después de este reporte): la fila no debe
    crashear, tiene que reportarlo como pendiente (None)."""
    fila = cb.fila_modelo(por_nombre("Qwen2.5-0.5B-Instruct"))
    assert fila["prompt_original"] is None
    assert fila["prompt_2026"] is not None
    assert fila["prompt_2026"]["exact_match_pct"] == 21.9
    assert fila["efecto_prompt_exact_match_pp"] is None


def test_matriz_incluye_los_cuatro_modelos_en_orden_de_registro():
    filas = cb.matriz()
    nombres = [f["modelo"] for f in filas]
    assert nombres == [m.nombre for m in roster_baseline_original()]
    assert len(nombres) == 4


# --------------------------------------------------------------------------
# Mejor bajo un prompt fijo / titular
# --------------------------------------------------------------------------

def test_mejor_bajo_prompt_se_calcula_por_codigo_no_se_hardcodea():
    filas = [
        {"modelo": "A", "prompt_2026": {"exact_match_pct": 10.0, "json_valido_pct": 100.0}},
        {"modelo": "B", "prompt_2026": {"exact_match_pct": 40.0, "json_valido_pct": 100.0}},
        {"modelo": "C", "prompt_2026": None},
    ]
    mejor = cb.mejor_bajo_prompt(filas, "prompt_2026")
    assert mejor == {"modelo": "B", "exact_match_pct": 40.0}


def test_mejor_bajo_prompt_devuelve_none_si_ninguna_fila_tiene_esa_celda():
    filas = [{"modelo": "A", "prompt_original": None}]
    assert cb.mejor_bajo_prompt(filas, "prompt_original") is None


def test_titular_con_datos_reales_compara_granite_4_0_1b_vs_mejor_generacion_anterior():
    """El titular del paper: mejor arquitectura 2026 (leída de su propio CSV
    real) vs mejor generación anterior, ambos bajo el prompt 2026 -- nunca se
    afirma a mano."""
    filas = cb.matriz()
    resultado = cb.titular(filas)
    assert resultado is not None
    assert resultado["prompt"] == "2026"
    assert resultado["mejor_arquitectura_2026"]["modelo"] == "granite-4.0-1b"
    assert resultado["mejor_arquitectura_2026"]["exact_match_pct"] == pytest.approx(90.6, abs=0.1)
    assert resultado["mejor_generacion_anterior"]["modelo"] == "Qwen2.5-1.5B-Instruct"
    assert resultado["mejor_generacion_anterior"]["exact_match_pct"] == pytest.approx(65.6, abs=0.1)
    assert resultado["delta_pp"] == pytest.approx(25.0, abs=0.1)


def test_titular_devuelve_none_si_todavia_no_hay_ninguna_celda_2026(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "DIR_DETALLE", tmp_path / "detalle")
    monkeypatch.setattr(cb, "DIR_BASELINE", tmp_path / "baseline_original")
    (tmp_path / "detalle").mkdir()
    (tmp_path / "baseline_original").mkdir()
    filas = cb.matriz()
    assert cb.titular(filas) is None


# --------------------------------------------------------------------------
# Guardia de sanidad: nunca comparar celdas de distinto prompt
# --------------------------------------------------------------------------

def test_comparar_celdas_rechaza_mezclar_prompts():
    fila_a = {"modelo": "A", "prompt_2026": {"exact_match_pct": 90.6}, "prompt_original": None}
    fila_b = {"modelo": "B", "prompt_2026": None,
              "prompt_original": {"exact_match_pct": 50.0}}
    with pytest.raises(ValueError, match="prompt"):
        cb.comparar_celdas(fila_a, "2026", fila_b, "original")


def test_comparar_celdas_calcula_el_delta_bajo_el_mismo_prompt():
    fila_a = {"modelo": "A", "prompt_2026": {"exact_match_pct": 90.6}, "prompt_original": None}
    fila_b = {"modelo": "B", "prompt_2026": {"exact_match_pct": 65.6}, "prompt_original": None}
    assert cb.comparar_celdas(fila_a, "2026", fila_b, "2026") == pytest.approx(25.0, abs=0.1)


def test_comparar_celdas_rechaza_una_celda_todavia_pendiente():
    fila_a = {"modelo": "A", "prompt_original": None}
    fila_b = {"modelo": "B", "prompt_original": {"exact_match_pct": 50.0}}
    with pytest.raises(ValueError):
        cb.comparar_celdas(fila_a, "original", fila_b, "original")


# --------------------------------------------------------------------------
# Ausencia de framing de auditoría / referencia al borrador
# --------------------------------------------------------------------------

def test_no_hay_lenguaje_de_auditoria_ni_de_baseline_publicado():
    """El módulo puede mencionar que NO es una auditoría (la corrección
    explícita es deseable); lo que no puede aparecer es la framing de
    "irreproducibilidad como aporte" ni tratar al borrador como un baseline
    validado contra el que comparar."""
    fuente = (RAIZ / "src" / "comparar_baseline.py").read_text(encoding="utf-8").lower()
    for termino_prohibido in (
        "irreproducibilidad es un aporte", "aporte secundario", "baseline validado",
        "publicado (hist", "cifra publicada",
    ):
        assert termino_prohibido not in fuente, termino_prohibido


def test_nota_latencia_advierte_sobre_no_comparabilidad():
    texto = cb.NOTA_LATENCIA.lower()
    assert "latencia" in texto
    assert "no" in texto
    assert "ocios" in texto  # ocioso/ociosa


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def test_escribir_comparacion_escribe_json_indentado(tmp_path):
    comparacion = {"matriz": [], "titular": None, "nota_latencia": "x"}
    salida = tmp_path / "out.json"
    cb.escribir_comparacion(comparacion, salida)
    assert json.loads(salida.read_text(encoding="utf-8")) == comparacion


def test_imprimir_tabla_no_crashea_con_celdas_faltantes(capsys):
    comparacion = {
        "matriz": [{
            "modelo": "X", "prompt_original": None, "prompt_2026": None,
            "efecto_prompt_exact_match_pp": None, "efecto_prompt_json_valido_pp": None,
        }],
        "titular": None,
        "nota_latencia": cb.NOTA_LATENCIA,
    }
    cb.imprimir_tabla(comparacion)
    salida = capsys.readouterr().out
    assert "X" in salida
    assert "pendiente" in salida.lower() or "—" in salida


def test_main_escribe_y_no_toca_el_repo(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cb, "SALIDA_PATH", tmp_path / "matriz_generacion_anterior.json")
    codigo = cb.main()
    assert codigo == 0
    assert (tmp_path / "matriz_generacion_anterior.json").exists()
    assert "modelo" in capsys.readouterr().out.lower()


def test_el_modulo_de_comparacion_baseline_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "comparar_baseline.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea
