---
id: 06
title: Consolidación de etapa 1 y selección del juez
depends_on: [05]
files:
  - src/stage1_2026.py
  - tests/test_stage1_2026.py
  - data/2026/detalle_2026.csv
  - data/2026/resumen_etapa1.json
  - data/2026/juez_seleccionado.json
---

## Spec

Cerrar la etapa 1: consolidar los 14 CSV por modelo en `data/2026/detalle_2026.csv`, calcular el resumen de etapa 1 y **elegir el juez de forma determinista** según **RF7** — mayor `exact_match_pct`; empate → mayor `params_b`; empate persistente → orden del roster.

Produce los tres artefactos de **F5**: `detalle_2026.csv`, `resumen_etapa1.json` y `juez_seleccionado.json`. Este último es el contrato de entrada del subtask 07: el juez no se elige a mano ni se hardcodea en ningún lado.

La reproducibilidad del pipeline **no** depende de fijar el ID del juez, sino de la decodificación greedy (RNF3); registrar el ganador es documentación, no garantía. Aun así la selección debe ser una función pura y total del resumen de etapa 1, para que dos corridas sobre los mismos datos elijan siempre lo mismo.

## Implementation plan

### Tarea 1 — Consolidación (TDD)

- [ ] Escribir el test que falla, `tests/test_stage1_2026.py` (primera parte):

```python
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
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_stage1_2026.py`
- [ ] Implementar la primera parte de `src/stage1_2026.py`:

```python
#!/usr/bin/env python3
"""
Cierre de la etapa 1: consolida los CSV por modelo, resume, y elige el juez.

La selección del juez es una función pura y total del resumen (RF7), para que
dos corridas sobre los mismos datos elijan siempre el mismo modelo. La
reproducibilidad de la etapa 2 la garantiza la decodificación greedy, no este
archivo; registrar al ganador es trazabilidad.

Uso:
    python src/stage1_2026.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from models_2026 import MODELOS_2026, por_nombre, slug
from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
DIR_DETALLE = DIR_2026 / "detalle"
PATH_CONSOLIDADO = DIR_2026 / "detalle_2026.csv"
PATH_RESUMEN_E1 = DIR_2026 / "resumen_etapa1.json"
PATH_JUEZ = DIR_2026 / "juez_seleccionado.json"


def consolidar_detalle(dir_detalle: Path, nombres: list[str]) -> pd.DataFrame:
    """Concatena los CSV por modelo en orden de roster; falla si falta alguno."""
    partes, faltantes = [], []
    for nombre in nombres:
        ruta = dir_detalle / f"{slug(nombre)}.csv"
        if not ruta.exists():
            ruta = dir_detalle / f"{nombre}.csv"   # tolerancia en tests
        if not ruta.exists():
            faltantes.append(nombre)
            continue
        df = pd.read_csv(ruta)
        if list(df.columns) != COLUMNAS_DETALLE:
            raise ValueError(f"{ruta} tiene un esquema distinto al congelado en F5")
        if len(df) != N_COMANDOS_ESPERADO:
            raise ValueError(f"{ruta} tiene {len(df)} filas, se esperaban {N_COMANDOS_ESPERADO}")
        partes.append(df.sort_values("idx"))
    if faltantes:
        raise ValueError(f"faltan CSV de detalle para: {', '.join(faltantes)}")
    return pd.concat(partes, ignore_index=True)[COLUMNAS_DETALLE]


def resumen_etapa1(df: pd.DataFrame, params_por_modelo: dict[str, float]) -> list[dict]:
    """Resumen previo a la etapa 2: sin métrica laxa, que aún no existe."""
    filas = []
    for nombre, params_b in params_por_modelo.items():
        df_m = df[df["modelo"] == nombre]
        if df_m.empty:
            continue
        filas.append({
            "modelo": nombre,
            "params_b": params_b,
            "n": len(df_m),
            "json_valido_pct": round(100 * df_m["json_valido"].mean(), 1),
            "exact_match_pct": round(100 * df_m["match_exact"].mean(), 1),
            "avg_latencia_s": round(df_m["latencia_s"].mean(), 3),
        })
    return filas
```

- [ ] Correr y confirmar **verde** la primera parte: `pytest -q tests/test_stage1_2026.py`

### Tarea 2 — Selección determinista del juez (TDD)

- [ ] Agregar a `tests/test_stage1_2026.py`:

```python
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
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_stage1_2026.py`
- [ ] Implementar `elegir_juez` y el `main`:

```python
CRITERIO_JUEZ = (
    "mayor exact_match_pct en etapa 1; desempate por mayor params_b; "
    "luego orden del roster"
)


def elegir_juez(resumen: list[dict]) -> dict:
    """Elige el juez de la etapa 2 (RF7). Función pura y total del resumen."""
    if not resumen:
        raise ValueError("El resumen de etapa 1 está vacío: no hay juez posible")

    orden_roster = {m.nombre: i for i, m in enumerate(MODELOS_2026)}
    mejor_pct = max(f["exact_match_pct"] for f in resumen)
    empatados = [f for f in resumen if f["exact_match_pct"] == mejor_pct]
    ganador = min(
        empatados,
        key=lambda f: (-f["params_b"], orden_roster.get(f["modelo"], len(orden_roster))),
    )
    return {
        "modelo": ganador["modelo"],
        "hf_repo_id": por_nombre(ganador["modelo"]).hf_repo_id,
        "params_b": ganador["params_b"],
        "exact_match_pct": ganador["exact_match_pct"],
        "criterio": CRITERIO_JUEZ,
        "empatados": sorted(f["modelo"] for f in empatados),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cierre de la etapa 1")
    parser.add_argument("--dir-detalle", default=str(DIR_DETALLE))
    args = parser.parse_args()

    nombres = [m.nombre for m in MODELOS_2026]
    df = consolidar_detalle(Path(args.dir_detalle), nombres)
    PATH_CONSOLIDADO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_CONSOLIDADO, index=False)

    resumen = resumen_etapa1(df, {m.nombre: m.params_b for m in MODELOS_2026})
    PATH_RESUMEN_E1.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    juez = elegir_juez(resumen)
    PATH_JUEZ.write_text(json.dumps(juez, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Consolidado: {len(df)} filas en {PATH_CONSOLIDADO}")
    for f in sorted(resumen, key=lambda x: -x["exact_match_pct"]):
        print(f"  {f['modelo']:26s} exact={f['exact_match_pct']:5.1f}%  "
              f"lat={f['avg_latencia_s']:7.2f}s")
    print(f"\nJuez elegido: {juez['modelo']} ({juez['exact_match_pct']}%), "
          f"empatados: {juez['empatados']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_stage1_2026.py`

### Tarea 3 — Ejecutar sobre los datos reales y commitear

- [ ] `python src/stage1_2026.py`
- [ ] Revisar a ojo el ranking impreso: ¿el ganador es plausible (suele ser de los más grandes)? Si ganara un sub-1B, no es un error — es un resultado y se discute en el paper.
- [ ] `pytest -q`
- [ ] `git add src/stage1_2026.py tests/test_stage1_2026.py data/2026/detalle_2026.csv data/2026/resumen_etapa1.json data/2026/juez_seleccionado.json`
- [ ] `git commit -m "feat(2026): consolidacion de etapa 1 y seleccion determinista del juez"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. El consolidado tiene las 448 filas, en orden de roster
python -c "
import sys, json; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import MODELOS_2026
from run_sweep_2026 import COLUMNAS_DETALLE
df = pd.read_csv('data/2026/detalle_2026.csv')
assert len(df) == 448, len(df)
assert list(df.columns) == COLUMNAS_DETALLE
assert df['modelo'].drop_duplicates().tolist() == [m.nombre for m in MODELOS_2026]
assert not df.duplicated(['modelo','idx']).any()
print('consolidado OK')
"

# 3. El juez elegido cumple el criterio, verificado independientemente
python -c "
import sys, json; sys.path.insert(0,'src')
from models_2026 import MODELOS_2026
resumen = json.load(open('data/2026/resumen_etapa1.json', encoding='utf-8'))
juez = json.load(open('data/2026/juez_seleccionado.json', encoding='utf-8'))
mejor = max(f['exact_match_pct'] for f in resumen)
emp = [f for f in resumen if f['exact_match_pct'] == mejor]
assert juez['exact_match_pct'] == mejor
assert juez['params_b'] == max(f['params_b'] for f in emp)
assert juez['modelo'] in {m.nombre for m in MODELOS_2026}
assert sorted(juez['empatados']) == sorted(f['modelo'] for f in emp)
print('juez:', juez['modelo'], juez['exact_match_pct'], '| empatados:', juez['empatados'])
"

# 4. Reejecutar es idempotente (byte a byte)
md5sum data/2026/juez_seleccionado.json data/2026/detalle_2026.csv > /tmp/antes.txt
python src/stage1_2026.py >/dev/null
md5sum -c /tmp/antes.txt && echo "idempotente OK"

# 5. F0 intacto
git diff --exit-code main -- data/resultados_experimento_detalle.csv \
  data/resultados_experimento_resumen.json tests/test_metricas.py && echo "F0 intacto"
```

## Acceptance criteria

- **Dado** los 14 CSV de detalle, **cuando** se corre `src/stage1_2026.py`, **entonces** `data/2026/detalle_2026.csv` tiene 448 filas, las 17 columnas de F5 en orden, los modelos en orden de roster, cada bloque con `idx` de 0 a 31, y sin pares `(modelo, idx)` duplicados.
- **Dado** un directorio de detalle al que le falta algún modelo del roster, **entonces** `consolidar_detalle` lanza `ValueError` nombrando los faltantes, en vez de producir un consolidado incompleto.
- **Dado** un CSV con esquema distinto o con distinto número de filas, **entonces** `consolidar_detalle` lanza `ValueError` señalando el archivo.
- **Dado** un resumen con un único máximo de `exact_match_pct`, **entonces** `elegir_juez` devuelve ese modelo y `empatados` contiene solo a él.
- **Dado** un empate en `exact_match_pct`, **entonces** gana el de mayor `params_b`; **dado** un empate también en `params_b`, **entonces** gana el que aparece antes en `MODELOS_2026`.
- **Dado** el mismo resumen en cualquier orden de entrada, **entonces** `elegir_juez` devuelve siempre el mismo modelo (determinismo total).
- **Dado** un resumen vacío, **entonces** `elegir_juez` lanza `ValueError`.
- **Dado** `juez_seleccionado.json`, **entonces** tiene exactamente las claves de F5 (`modelo`, `hf_repo_id`, `params_b`, `exact_match_pct`, `criterio`, `empatados`), su `modelo` pertenece al roster, su `hf_repo_id` coincide con el del registro, y el `exact_match_pct` es efectivamente el máximo del `resumen_etapa1.json`.
- **Dado** que el script se corre dos veces seguidas, **entonces** los tres artefactos quedan byte-idénticos.
- **Dado** el commit, **entonces** `pytest -q` pasa y los archivos de F0 siguen intactos.
