---
id: 09
title: "Métricas 2026: estricta, laxa, Tablas 2/3/4"
depends_on: [08]
files:
  - src/metrics_2026.py
  - tests/test_metrics_2026.py
  - data/2026/resumen_2026.json
  - data/2026/taxonomia_2026.csv
  - data/2026/exactitud_por_categoria.csv
---

## Spec

Implementar **F6**: las tres funciones de agregación que producen las tablas del paper reescrito, y ejecutarlas sobre los datos reales.

- **RF10** — dos métricas titulares por modelo: **estricta** (`match_exact`) y **laxa** (`match_exact` o etiqueta en `ETIQUETAS_EQUIVALENTES`). La brecha entre ambas es el resultado central que responde a la amenaza de validez de constructo del paper original.
- **RF11** — Tabla 4 con las **7** etiquetas, con `alucinacion_valor_unidad` y `valor_numerico_incorrecto` **separadas** (el legacy `metrics.py` las fusiona en `confusion_valor_o_unidad`; ese módulo queda congelado y no se toca).
- **RF9 / Tabla 3** — exactitud estricta por categoría lingüística × modelo, usando las categorías que produjo el juez, no una columna del dataset.

Produce `resumen_2026.json` (esquema exacto de F5), `taxonomia_2026.csv` y `exactitud_por_categoria.csv`.

## Implementation plan

### Tarea 1 — Resumen con métrica estricta y laxa (TDD)

- [ ] Escribir el test que falla, `tests/test_metrics_2026.py` (primera parte):

```python
"""Tests de las métricas 2026: estricta vs laxa, taxonomía de 7, Tabla 3 (F6)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics_2026 import (  # noqa: E402
    CLAVES_RESUMEN,
    calcular_exactitud_por_categoria,
    calcular_resumen_2026,
    calcular_taxonomia_errores_2026,
)
from models_2026 import MODELOS_2026  # noqa: E402
from run_sweep_2026 import COLUMNAS_DETALLE  # noqa: E402
from taxonomia_2026 import ETIQUETAS_ERROR  # noqa: E402

M1, M2 = MODELOS_2026[0].nombre, MODELOS_2026[1].nombre


def _det(modelo, patron, latencia=1.0, modo="chat_template"):
    """patron: lista de bools de match_exact, una por comando."""
    filas = []
    for i, ok in enumerate(patron):
        filas.append({
            "modelo": modelo, "idx": i, "comando": f"c{i}", "gt": "{}",
            "pred_raw": "{}", "pred_json": "{}", "json_valido": True,
            "parse_note": "", "latencia_s": latencia,
            "match_intent": ok, "match_dispositivo": ok, "match_ubicacion": ok,
            "match_valor": ok, "match_unidad": ok, "match_exact": ok,
            "modo_prompting": modo, "transformers_version": "4.57.0",
        })
    return pd.DataFrame(filas, columns=COLUMNAS_DETALLE)


def _eti(pares):
    """pares: lista de (modelo, idx, etiquetas)."""
    return pd.DataFrame(
        [{"modelo": m, "idx": i, "etiquetas": e, "juez_raw": e, "juez_parse_ok": True}
         for m, i, e in pares],
        columns=["modelo", "idx", "etiquetas", "juez_raw", "juez_parse_ok"],
    )


def test_resumen_tiene_las_claves_congeladas_y_en_orden():
    det = _det(M1, [True, False, True, False])
    eti = _eti([(M1, 1, "confusion_intencion"), (M1, 3, "uso_de_sinonimos")])
    filas = calcular_resumen_2026(det, eti)
    assert len(filas) == 1
    assert list(filas[0].keys()) == CLAVES_RESUMEN


def test_la_laxa_rescata_sinonimos_y_equivalentes():
    det = _det(M1, [True, False, False, False])
    eti = _eti([
        (M1, 1, "uso_de_sinonimos"),
        (M1, 2, "sin_error_semantico"),
        (M1, 3, "confusion_intencion"),
    ])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_pct"] == 25.0        # 1 de 4
    assert fila["exact_match_laxo_pct"] == 75.0   # 1 + 2 rescatadas


def test_la_laxa_rescata_aunque_la_etiqueta_venga_combinada():
    det = _det(M1, [False])
    eti = _eti([(M1, 0, "confusion_ubicacion;uso_de_sinonimos")])
    assert calcular_resumen_2026(det, eti)[0]["exact_match_laxo_pct"] == 100.0


def test_la_laxa_nunca_es_menor_que_la_estricta():
    det = pd.concat([_det(M1, [True, False] * 8), _det(M2, [False] * 16)])
    eti = _eti([(M1, i, "confusion_intencion") for i in range(1, 16, 2)]
               + [(M2, i, "sin_error_semantico") for i in range(16)])
    for fila in calcular_resumen_2026(det, eti):
        assert fila["exact_match_laxo_pct"] >= fila["exact_match_pct"]
        assert 0.0 <= fila["exact_match_pct"] <= 100.0
        assert 0.0 <= fila["exact_match_laxo_pct"] <= 100.0


def test_sin_etiquetas_de_equivalencia_ambas_metricas_coinciden():
    det = _det(M1, [True, False])
    eti = _eti([(M1, 1, "confusion_dispositivo")])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_laxo_pct"] == fila["exact_match_pct"] == 50.0


def test_el_resumen_trae_metadatos_del_registro():
    det = _det(M1, [True], modo="raw_completion")
    fila = calcular_resumen_2026(det, _eti([]))[0]
    assert fila["hf_repo_id"] == MODELOS_2026[0].hf_repo_id
    assert fila["tier"] == MODELOS_2026[0].tier
    assert fila["transformers_pin"] == MODELOS_2026[0].transformers_pin
    assert fila["modo_prompting"] == "raw_completion"


def test_el_resumen_respeta_el_orden_del_roster():
    det = pd.concat([_det(M2, [True]), _det(M1, [True])])  # al revés a propósito
    assert [f["modelo"] for f in calcular_resumen_2026(det, _eti([]))] == [M1, M2]
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_metrics_2026.py`
- [ ] Implementar la primera parte de `src/metrics_2026.py`:

```python
#!/usr/bin/env python3
"""
Agregación de métricas 2026.

Módulo nuevo, no una edición de metrics.py: el legacy está congelado porque
tests/test_metricas.py fija contra él los números publicados del paper
original. Dos diferencias de fondo con aquel: acá hay dos métricas de
exactitud (estricta y laxa) en vez de una, y la taxonomía tiene 7 etiquetas
con alucinacion_valor_unidad y valor_numerico_incorrecto separadas, como en
la Tabla 4 publicada (el legacy las fusiona).

Uso:
    python src/metrics_2026.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from models_2026 import MODELOS_2026
from taxonomia_2026 import (
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_EQUIVALENTES,
    ETIQUETAS_ERROR,
    SEP_ETIQUETAS,
)

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
PATH_DETALLE = DIR_2026 / "detalle_2026.csv"
PATH_ETIQUETAS = DIR_2026 / "etiquetas_errores.csv"
PATH_CATEGORIAS = DIR_2026 / "categorias_comandos.csv"
PATH_RESUMEN = DIR_2026 / "resumen_2026.json"
PATH_TAXONOMIA = DIR_2026 / "taxonomia_2026.csv"
PATH_POR_CATEGORIA = DIR_2026 / "exactitud_por_categoria.csv"

CLAVES_RESUMEN = [
    "modelo", "hf_repo_id", "params_b", "tier", "modo_prompting",
    "transformers_pin", "n", "json_valido_pct", "exact_match_pct",
    "exact_match_laxo_pct", "avg_latencia_s", "acc_intent_pct",
    "acc_dispositivo_pct", "acc_ubicacion_pct", "acc_valor_pct",
    "acc_unidad_pct",
]


def _mapa_equivalentes(df_etiquetas: pd.DataFrame) -> set[tuple[str, int]]:
    """(modelo, idx) que el juez consideró semánticamente equivalentes."""
    equivalentes = set()
    for _, fila in df_etiquetas.iterrows():
        partes = set(str(fila["etiquetas"]).split(SEP_ETIQUETAS))
        if partes & ETIQUETAS_EQUIVALENTES:
            equivalentes.add((fila["modelo"], int(fila["idx"])))
    return equivalentes


def calcular_resumen_2026(df_detalle: pd.DataFrame,
                          df_etiquetas: pd.DataFrame) -> list[dict]:
    """Resumen por modelo con exactitud estricta y laxa (RF10)."""
    equivalentes = _mapa_equivalentes(df_etiquetas)
    filas = []
    for modelo in MODELOS_2026:
        df_m = df_detalle[df_detalle["modelo"] == modelo.nombre]
        if df_m.empty:
            continue
        estricta = df_m["match_exact"].astype(bool)
        laxa = estricta | df_m["idx"].map(
            lambda i: (modelo.nombre, int(i)) in equivalentes
        )
        modos = df_m["modo_prompting"].unique()
        filas.append({
            "modelo": modelo.nombre,
            "hf_repo_id": modelo.hf_repo_id,
            "params_b": modelo.params_b,
            "tier": modelo.tier,
            "modo_prompting": modos[0] if len(modos) == 1 else "mixto",
            "transformers_pin": modelo.transformers_pin,
            "n": len(df_m),
            "json_valido_pct": round(100 * df_m["json_valido"].astype(bool).mean(), 1),
            "exact_match_pct": round(100 * estricta.mean(), 1),
            "exact_match_laxo_pct": round(100 * laxa.mean(), 1),
            "avg_latencia_s": round(df_m["latencia_s"].mean(), 3),
            "acc_intent_pct": round(100 * df_m["match_intent"].astype(bool).mean(), 1),
            "acc_dispositivo_pct": round(100 * df_m["match_dispositivo"].astype(bool).mean(), 1),
            "acc_ubicacion_pct": round(100 * df_m["match_ubicacion"].astype(bool).mean(), 1),
            "acc_valor_pct": round(100 * df_m["match_valor"].astype(bool).mean(), 1),
            "acc_unidad_pct": round(100 * df_m["match_unidad"].astype(bool).mean(), 1),
        })
    return [{c: f[c] for c in CLAVES_RESUMEN} for f in filas]
```

- [ ] Correr y confirmar **verde** la primera parte: `pytest -q tests/test_metrics_2026.py`

### Tarea 2 — Taxonomía de 7 etiquetas y Tabla 3 (TDD)

- [ ] Agregar a `tests/test_metrics_2026.py`:

```python
def test_taxonomia_tiene_las_siete_columnas_separadas():
    det = _det(M1, [False, False])
    eti = _eti([(M1, 0, "alucinacion_valor_unidad"), (M1, 1, "valor_numerico_incorrecto")])
    tax = calcular_taxonomia_errores_2026(det, eti)
    assert list(tax.columns) == ["modelo", "total_incorrectas"] + ETIQUETAS_ERROR
    fila = tax.iloc[0]
    assert fila["alucinacion_valor_unidad"] == 1
    assert fila["valor_numerico_incorrecto"] == 1
    assert "confusion_valor_o_unidad" not in tax.columns   # el legacy las fusionaba


def test_taxonomia_cuenta_etiquetas_no_excluyentes():
    det = _det(M1, [False, False, False])
    eti = _eti([
        (M1, 0, "confusion_intencion;confusion_dispositivo"),
        (M1, 1, "confusion_intencion"),
        (M1, 2, "uso_de_sinonimos"),
    ])
    fila = calcular_taxonomia_errores_2026(det, eti).iloc[0]
    assert fila["total_incorrectas"] == 3
    assert fila["confusion_intencion"] == 2
    assert fila["confusion_dispositivo"] == 1
    assert fila["uso_de_sinonimos"] == 1
    assert fila["confusion_ubicacion"] == 0


def test_taxonomia_incluye_una_fila_por_modelo_en_orden_de_roster():
    det = pd.concat([_det(M2, [False]), _det(M1, [False])])
    eti = _eti([(M1, 0, "confusion_intencion"), (M2, 0, "confusion_ubicacion")])
    tax = calcular_taxonomia_errores_2026(det, eti)
    assert tax["modelo"].tolist() == [M1, M2]


def test_exactitud_por_categoria_agrupa_bien():
    # idx 0,1 -> encendido simple ; idx 2,3 -> ajuste con valor
    det = _det(M1, [True, False, True, True])
    cat = pd.DataFrame([
        {"idx": 0, "comando": "c0", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 1, "comando": "c1", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 2, "comando": "c2", "categoria": "ajuste_con_valor_numerico",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 3, "comando": "c3", "categoria": "ajuste_con_valor_numerico",
         "juez_raw": "", "juez_parse_ok": True},
    ])
    tabla = calcular_exactitud_por_categoria(det, cat)
    assert list(tabla.columns) == ["categoria", "n", M1]
    por_cat = tabla.set_index("categoria")
    assert por_cat.loc["encendido_apagado_simple", "n"] == 2
    assert por_cat.loc["encendido_apagado_simple", M1] == 50.0
    assert por_cat.loc["ajuste_con_valor_numerico", M1] == 100.0


def test_exactitud_por_categoria_omite_categorias_sin_comandos():
    det = _det(M1, [True])
    cat = pd.DataFrame([{"idx": 0, "comando": "c0", "categoria": "consulta_de_estado",
                         "juez_raw": "", "juez_parse_ok": True}])
    tabla = calcular_exactitud_por_categoria(det, cat)
    assert tabla["categoria"].tolist() == ["consulta_de_estado"]


def test_exactitud_por_categoria_respeta_el_orden_canonico():
    det = _det(M1, [True, True])
    cat = pd.DataFrame([
        {"idx": 0, "comando": "c0", "categoria": "consulta_de_estado",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 1, "comando": "c1", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
    ])
    tabla = calcular_exactitud_por_categoria(det, cat)
    # encendido_apagado_simple va antes que consulta_de_estado en el vocabulario
    assert tabla["categoria"].tolist() == ["encendido_apagado_simple", "consulta_de_estado"]
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_metrics_2026.py`
- [ ] Implementar el resto de `src/metrics_2026.py`:

```python
def calcular_taxonomia_errores_2026(df_detalle: pd.DataFrame,
                                    df_etiquetas: pd.DataFrame) -> pd.DataFrame:
    """Tabla 4 con las 7 etiquetas, no excluyentes (RF11)."""
    filas = []
    for modelo in MODELOS_2026:
        df_m = df_detalle[df_detalle["modelo"] == modelo.nombre]
        if df_m.empty:
            continue
        incorrectas = df_m[~df_m["match_exact"].astype(bool)]
        eti_m = df_etiquetas[df_etiquetas["modelo"] == modelo.nombre]
        conteos = {e: 0 for e in ETIQUETAS_ERROR}
        for valor in eti_m["etiquetas"]:
            for etiqueta in str(valor).split(SEP_ETIQUETAS):
                if etiqueta in conteos:
                    conteos[etiqueta] += 1
        filas.append({
            "modelo": modelo.nombre,
            "total_incorrectas": len(incorrectas),
            **conteos,
        })
    return pd.DataFrame(filas, columns=["modelo", "total_incorrectas"] + ETIQUETAS_ERROR)


def calcular_exactitud_por_categoria(df_detalle: pd.DataFrame,
                                     df_categorias: pd.DataFrame) -> pd.DataFrame:
    """Tabla 3: exactitud estricta por categoría lingüística x modelo (RF9)."""
    categoria_por_idx = dict(
        zip(df_categorias["idx"].astype(int), df_categorias["categoria"])
    )
    df = df_detalle.copy()
    df["categoria"] = df["idx"].astype(int).map(categoria_por_idx)

    presentes = [c for c in CATEGORIAS_LINGUISTICAS if c in set(df["categoria"])]
    modelos = [m.nombre for m in MODELOS_2026 if m.nombre in set(df["modelo"])]

    filas = []
    for categoria in presentes:
        df_c = df[df["categoria"] == categoria]
        fila = {"categoria": categoria, "n": int(df_c["idx"].nunique())}
        for nombre in modelos:
            df_cm = df_c[df_c["modelo"] == nombre]
            fila[nombre] = (
                round(100 * df_cm["match_exact"].astype(bool).mean(), 1)
                if not df_cm.empty else 0.0
            )
        filas.append(fila)
    return pd.DataFrame(filas, columns=["categoria", "n"] + modelos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Métricas 2026")
    parser.add_argument("--detalle", default=str(PATH_DETALLE))
    parser.add_argument("--etiquetas", default=str(PATH_ETIQUETAS))
    parser.add_argument("--categorias", default=str(PATH_CATEGORIAS))
    args = parser.parse_args()

    det = pd.read_csv(args.detalle)
    eti = pd.read_csv(args.etiquetas)
    cat = pd.read_csv(args.categorias)

    resumen = calcular_resumen_2026(det, eti)
    PATH_RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    PATH_RESUMEN.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calcular_taxonomia_errores_2026(det, eti).to_csv(PATH_TAXONOMIA, index=False)
    calcular_exactitud_por_categoria(det, cat).to_csv(PATH_POR_CATEGORIA, index=False)

    print(f"{'modelo':28s} {'estricta':>9s} {'laxa':>7s} {'brecha':>7s} {'lat(s)':>8s}")
    for f in resumen:
        brecha = round(f["exact_match_laxo_pct"] - f["exact_match_pct"], 1)
        print(f"{f['modelo']:28s} {f['exact_match_pct']:8.1f}% "
              f"{f['exact_match_laxo_pct']:6.1f}% {brecha:6.1f}pp "
              f"{f['avg_latencia_s']:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_metrics_2026.py`

### Tarea 3 — Ejecutar sobre los datos reales y commitear

- [ ] `python src/metrics_2026.py`
- [ ] Revisar la columna "brecha": es el resultado central del paper. Si fuera 0.0 para todos los modelos, verificar el subtask 08 antes de seguir.
- [ ] `pytest -q`
- [ ] `git add src/metrics_2026.py tests/test_metrics_2026.py data/2026/resumen_2026.json data/2026/taxonomia_2026.csv data/2026/exactitud_por_categoria.csv`
- [ ] `git commit -m "feat(2026): metricas estricta y laxa, taxonomia de 7 etiquetas y tabla por categoria"`

## Verify

```bash
# 1. Suite verde, incluida la regresión legacy sin tocar metrics.py
pytest -q
git diff --exit-code main -- src/metrics.py tests/test_metricas.py && echo "legacy intacto"

# 2. resumen_2026.json cumple el esquema y los invariantes de F5/F6
python -c "
import sys, json; sys.path.insert(0,'src')
from metrics_2026 import CLAVES_RESUMEN
from models_2026 import MODELOS_2026
r = json.load(open('data/2026/resumen_2026.json', encoding='utf-8'))
assert len(r) == 12, len(r)
assert [f['modelo'] for f in r] == [m.nombre for m in MODELOS_2026]
for f in r:
    assert list(f.keys()) == CLAVES_RESUMEN, f['modelo']
    assert f['n'] == 32
    assert 0 <= f['exact_match_pct'] <= 100
    assert f['exact_match_laxo_pct'] >= f['exact_match_pct'], f['modelo']
print('resumen OK; brechas:',
      [round(f['exact_match_laxo_pct']-f['exact_match_pct'],1) for f in r])
"

# 3. La taxonomía separa las dos categorías que el legacy fusionaba
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from taxonomia_2026 import ETIQUETAS_ERROR
t = pd.read_csv('data/2026/taxonomia_2026.csv')
assert list(t.columns) == ['modelo','total_incorrectas'] + ETIQUETAS_ERROR
assert 'alucinacion_valor_unidad' in t.columns and 'valor_numerico_incorrecto' in t.columns
assert 'confusion_valor_o_unidad' not in t.columns
assert len(t) == 12
print(t.to_string(index=False))
"

# 4. Tabla 3 suma 32 comandos y cubre solo categorías del vocabulario
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from taxonomia_2026 import CATEGORIAS_LINGUISTICAS
c = pd.read_csv('data/2026/exactitud_por_categoria.csv')
assert c['n'].sum() == 32, c['n'].sum()
assert c['categoria'].isin(CATEGORIAS_LINGUISTICAS).all()
assert c['categoria'].tolist() == [x for x in CATEGORIAS_LINGUISTICAS if x in set(c['categoria'])]
print(c.to_string(index=False))
"

# 5. Idempotencia
md5sum data/2026/resumen_2026.json data/2026/taxonomia_2026.csv \
       data/2026/exactitud_por_categoria.csv > /tmp/m_antes.txt
python src/metrics_2026.py >/dev/null && md5sum -c /tmp/m_antes.txt && echo "idempotente OK"
```

## Acceptance criteria

- **Dado** `calcular_resumen_2026`, **entonces** devuelve una fila por modelo del roster presente en el detalle, en orden de roster, con exactamente las 16 claves de `CLAVES_RESUMEN` en ese orden.
- **Dado** una fila incorrecta etiquetada `uso_de_sinonimos` o `sin_error_semantico` (sola o combinada con otras), **entonces** cuenta como acierto en la métrica laxa y **no** en la estricta.
- **Dado** cualquier modelo, **entonces** `exact_match_laxo_pct >= exact_match_pct` y ambas están en `[0, 100]`; si no hay ninguna etiqueta de equivalencia, las dos métricas coinciden exactamente.
- **Dado** el resumen, **entonces** cada fila trae `hf_repo_id`, `tier` y `transformers_pin` tomados del registro `models_2026`, y `modo_prompting` tomado del detalle (`"mixto"` si un modelo tuviera más de uno).
- **Dado** `calcular_taxonomia_errores_2026`, **entonces** sus columnas son `modelo, total_incorrectas` más **las 7** etiquetas, con `alucinacion_valor_unidad` y `valor_numerico_incorrecto` como columnas **separadas**, y sin ninguna columna `confusion_valor_o_unidad`.
- **Dado** una fila con etiquetas combinadas, **entonces** suma 1 a cada etiqueta presente: los conteos son no excluyentes y pueden sumar más que `total_incorrectas`.
- **Dado** `calcular_exactitud_por_categoria`, **entonces** devuelve `categoria, n` más una columna por modelo en orden de roster; `n` es la cantidad de **comandos distintos** de esa categoría; la suma de `n` es 32; las categorías aparecen en el orden canónico de `CATEGORIAS_LINGUISTICAS`, omitiendo las vacías.
- **Dado** `python src/metrics_2026.py` corrido dos veces, **entonces** los tres artefactos quedan byte-idénticos.
- **Dado** el commit, **entonces** `src/metrics.py` y `tests/test_metricas.py` siguen byte-idénticos a `main` y `pytest -q` pasa.
