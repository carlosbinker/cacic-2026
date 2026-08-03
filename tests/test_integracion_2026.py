"""AC2/AC3/AC4: integridad cross-task de `data/2026/`. Solo lee artefactos ya
producidos; nunca reejecuta el barrido ni al juez."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from models_2026 import MODELOS_2026  # noqa: E402
from taxonomia_2026 import (  # noqa: E402
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_ERROR,
    SEP_ETIQUETAS,
)

DIR_2026 = RAIZ / "data" / "2026"
DETALLE = DIR_2026 / "detalle_2026.csv"
ETIQUETAS = DIR_2026 / "etiquetas_errores.csv"
CATEGORIAS = DIR_2026 / "categorias_comandos.csv"
RESUMEN = DIR_2026 / "resumen_2026.json"

N_COMANDOS = 32
N_FILAS = len(MODELOS_2026) * N_COMANDOS  # 448

requiere_barrido = pytest.mark.skipif(
    not DETALLE.exists(), reason=f"falta {DETALLE.name}: requiere el subtask 06"
)
requiere_etapa2 = pytest.mark.skipif(
    not (ETIQUETAS.exists() and CATEGORIAS.exists()),
    reason="faltan los CSV de la etapa 2: requiere el subtask 08",
)
requiere_resumen = pytest.mark.skipif(
    not RESUMEN.exists(), reason=f"falta {RESUMEN.name}: requiere el subtask 09"
)


@requiere_barrido
def test_el_consolidado_tiene_448_filas_del_roster_completo():
    df = pd.read_csv(DETALLE)
    assert len(df) == N_FILAS
    assert sorted(df["modelo"].unique()) == sorted(m.nombre for m in MODELOS_2026)
    for nombre, grupo in df.groupby("modelo"):
        assert sorted(grupo["idx"]) == list(range(N_COMANDOS)), f"{nombre}: idx incompletos o duplicados"


@requiere_barrido
@requiere_etapa2
def test_la_etapa_2_cubre_exactamente_las_respuestas_incorrectas():
    df = pd.read_csv(DETALLE)
    etq = pd.read_csv(ETIQUETAS)
    incorrectas = {
        (f.modelo, f.idx) for f in df[~df["match_exact"].astype(bool)].itertuples()
    }
    etiquetadas = {(f.modelo, f.idx) for f in etq.itertuples()}
    assert incorrectas == etiquetadas
    for texto in etq["etiquetas"]:
        partes = [p for p in str(texto).split(SEP_ETIQUETAS) if p]
        assert partes, "hay una fila etiquetada con el conjunto vacío"
        assert set(partes) <= set(ETIQUETAS_ERROR), f"etiqueta fuera de vocabulario: {texto!r}"


@requiere_etapa2
def test_las_categorias_cubren_los_32_comandos_una_sola_vez():
    cat = pd.read_csv(CATEGORIAS)
    assert sorted(cat["idx"]) == list(range(N_COMANDOS))
    assert set(cat["categoria"]) <= set(CATEGORIAS_LINGUISTICAS)


@requiere_resumen
def test_la_exactitud_laxa_domina_a_la_estricta_en_los_14_modelos():
    resumen = json.loads(RESUMEN.read_text(encoding="utf-8"))
    assert [f["modelo"] for f in resumen] == [m.nombre for m in MODELOS_2026]
    for fila in resumen:
        estricta, laxa = fila["exact_match_pct"], fila["exact_match_laxo_pct"]
        assert 0 <= estricta <= 100, f"{fila['modelo']}: estricta fuera de rango"
        assert 0 <= laxa <= 100, f"{fila['modelo']}: laxa fuera de rango"
        assert laxa >= estricta, f"{fila['modelo']}: laxa {laxa} < estricta {estricta}"
