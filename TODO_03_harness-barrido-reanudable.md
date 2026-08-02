---
id: 03
title: Harness de barrido reanudable por modelo
depends_on: [01, 02]
files:
  - src/run_sweep_2026.py
  - tests/test_run_sweep_2026.py
---

## Spec

Implementar el harness que corre **un solo modelo por invocación** y escribe `data/2026/detalle/<slug>.csv` con las 17 columnas congeladas en **F5**. Es la pieza que se ejecuta dentro de cada contenedor del subtask 04.

Cubre **RF3** (reanudable: si el CSV del modelo ya existe y está completo, se saltea salvo `--force`), **RF6** (etapa 1 = coincidencia textual, reutilizando `scoring.py` sin tocarlo) y **RNF3** (decodificación greedy determinista, idéntica al harness legacy).

Diferencias respecto de `src/run_evaluation.py` (que queda congelado por F0): un modelo por corrida en vez de los cuatro, salida por modelo en vez de un CSV único, entrada construida por `prompt_2026.construir_entrada` en vez de `apply_chat_template` directo, y dos columnas nuevas (`modo_prompting`, `transformers_version`).

La lógica pura (armado de fila, decisión de saltear) se separa de la inferencia para poder testearla sin `torch` ni descargas.

## Implementation plan

### Tarea 1 — Lógica pura de reanudación y armado de fila (TDD)

- [ ] Escribir el test que falla, `tests/test_run_sweep_2026.py`:

```python
"""Tests de la lógica pura del harness de barrido 2026 (F5 del índice).

No se ejerce inferencia real: se testea el contrato de columnas, el armado
de filas y la decisión de reanudación, que es lo que puede romperse en
silencio durante un barrido de varias horas.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import por_nombre  # noqa: E402
from run_sweep_2026 import (  # noqa: E402
    COLUMNAS_DETALLE,
    N_COMANDOS_ESPERADO,
    armar_fila,
    debe_saltear,
    ruta_detalle,
)


def test_columnas_congeladas_en_orden():
    assert COLUMNAS_DETALLE == [
        "modelo", "idx", "comando", "gt", "pred_raw", "pred_json",
        "json_valido", "parse_note", "latencia_s",
        "match_intent", "match_dispositivo", "match_ubicacion",
        "match_valor", "match_unidad", "match_exact",
        "modo_prompting", "transformers_version",
    ]


def test_las_quince_primeras_columnas_son_las_del_csv_legacy():
    legacy = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "resultados_experimento_detalle.csv",
        nrows=1,
    )
    assert COLUMNAS_DETALLE[:15] == list(legacy.columns)


def test_ruta_detalle_usa_el_slug():
    ruta = ruta_detalle("Qwen3.5-0.8B")
    assert ruta.name == "qwen3-5-0-8b.csv"
    assert ruta.parent.name == "detalle" and ruta.parent.parent.name == "2026"


def _csv_completo(tmp_path: Path, n: int) -> Path:
    ruta = tmp_path / "m.csv"
    pd.DataFrame(
        {c: [None] * n for c in COLUMNAS_DETALLE} | {"idx": list(range(n))}
    ).to_csv(ruta, index=False)
    return ruta


def test_debe_saltear_si_el_csv_esta_completo(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO), force=False) is True


def test_no_saltea_si_falta_alguna_fila(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO - 1), force=False) is False


def test_no_saltea_si_no_existe(tmp_path):
    assert debe_saltear(tmp_path / "no-existe.csv", force=False) is False


def test_force_nunca_saltea(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO), force=True) is False


def test_no_saltea_si_el_csv_esta_corrupto(tmp_path):
    """Un barrido interrumpido a mitad de escritura no debe darse por bueno."""
    ruta = tmp_path / "corrupto.csv"
    ruta.write_text("modelo,idx\nesto,no,tiene,el,esquema\n", encoding="utf-8")
    assert debe_saltear(ruta, force=False) is False


def test_armar_fila_produce_exactamente_las_columnas_y_los_tipos():
    modelo = por_nombre("SmolLM2-360M-Instruct")
    gt = {"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo,
        idx=0,
        comando="Prendé la luz del living.",
        gt=gt,
        texto_generado='{"intent": "encender", "dispositivo": "luz", '
                       '"ubicacion": "living", "valor": null, "unidad": null}',
        latencia_s=1.23456,
        modo="chat_template",
        transformers_version="4.57.0",
    )
    assert list(fila.keys()) == COLUMNAS_DETALLE
    assert fila["modelo"] == "SmolLM2-360M-Instruct"
    assert fila["json_valido"] is True
    assert fila["match_exact"] is True
    assert fila["latencia_s"] == 1.235
    assert fila["modo_prompting"] == "chat_template"
    assert fila["transformers_version"] == "4.57.0"


def test_armar_fila_marca_incorrecta_una_respuesta_con_sinonimo():
    """'tele' en lugar de 'tv' es incorrecta en la etapa 1 (la laxa la rescata luego)."""
    modelo = por_nombre("SmolLM2-360M-Instruct")
    gt = {"intent": "apagar", "dispositivo": "tv", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo, idx=5, comando="Apagá la tele del living.", gt=gt,
        texto_generado='{"intent": "apagar", "dispositivo": "tele", '
                       '"ubicacion": "living", "valor": null, "unidad": null}',
        latencia_s=2.0, modo="chat_template", transformers_version="4.57.0",
    )
    assert fila["json_valido"] is True
    assert fila["match_dispositivo"] is False
    assert fila["match_exact"] is False


def test_armar_fila_con_salida_no_parseable():
    modelo = por_nombre("LFM2.5-230M")
    gt = {"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo, idx=1, comando="Prendé la luz.", gt=gt,
        texto_generado="No entiendo el comando.", latencia_s=0.5,
        modo="raw_completion", transformers_version="4.57.0",
    )
    assert fila["json_valido"] is False
    assert fila["pred_json"] == ""
    assert fila["parse_note"] == "no_parseable_como_json"
    assert fila["match_exact"] is False
    assert fila["modo_prompting"] == "raw_completion"


def test_armar_fila_rechaza_un_modo_invalido():
    modelo = por_nombre("LFM2.5-230M")
    with pytest.raises(ValueError, match="modo_prompting"):
        armar_fila(
            modelo=modelo, idx=1, comando="x", gt={}, texto_generado="{}",
            latencia_s=0.1, modo="inventado", transformers_version="4.57.0",
        )
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_run_sweep_2026.py`
- [ ] Implementar `src/run_sweep_2026.py`. Parte pura (sin `torch` a nivel de módulo — los imports pesados van dentro de `evaluar_modelo`, así los tests corren sin `transformers` instalado):

```python
#!/usr/bin/env python3
"""
Harness de barrido 2026: evalúa UN modelo del roster sobre los 32 comandos y
escribe data/2026/detalle/<slug>.csv.

Un modelo por invocación, a propósito: cada uno corre en su propia imagen
Docker (subtask 04) con su propia versión de transformers, y el barrido
completo debe ser reanudable --- son ~4-8 h de CPU y una interrupción no
puede obligar a rehacer los modelos ya terminados.

Uso:
    python src/run_sweep_2026.py --modelo "Qwen3.5-0.8B" [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from models_2026 import MODELOS_2026, ModeloEvaluado2026, por_nombre, slug
from prompt_2026 import construir_entrada
from scoring import comparar_campos, extraer_json, fila_a_ground_truth

RAIZ = Path(__file__).resolve().parent.parent
DATASET_PATH = RAIZ / "data" / "dataset_comandos_domotica.csv"
DIR_DETALLE = RAIZ / "data" / "2026" / "detalle"

N_COMANDOS_ESPERADO = 32
MAX_NEW_TOKENS = 128

COLUMNAS_DETALLE = [
    "modelo", "idx", "comando", "gt", "pred_raw", "pred_json",
    "json_valido", "parse_note", "latencia_s",
    "match_intent", "match_dispositivo", "match_ubicacion",
    "match_valor", "match_unidad", "match_exact",
    "modo_prompting", "transformers_version",
]

MODOS_VALIDOS = ("chat_template", "raw_completion")


def ruta_detalle(nombre_modelo: str) -> Path:
    return DIR_DETALLE / f"{slug(nombre_modelo)}.csv"


def debe_saltear(ruta: Path, force: bool) -> bool:
    """True si ese modelo ya tiene un CSV completo y bien formado."""
    if force or not ruta.exists():
        return False
    try:
        df = pd.read_csv(ruta)
    except Exception:
        return False
    if list(df.columns) != COLUMNAS_DETALLE:
        return False
    return len(df) == N_COMANDOS_ESPERADO


def armar_fila(*, modelo: ModeloEvaluado2026, idx: int, comando: str, gt: dict,
               texto_generado: str, latencia_s: float, modo: str,
               transformers_version: str) -> dict:
    """Arma una fila del CSV de detalle con las 17 columnas de F5, en orden."""
    if modo not in MODOS_VALIDOS:
        raise ValueError(
            f"modo_prompting inválido: {modo!r}. Válidos: {MODOS_VALIDOS}"
        )
    pred, json_valido, nota = extraer_json(texto_generado)
    matches = comparar_campos(pred, gt)
    fila = {
        "modelo": modelo.nombre,
        "idx": idx,
        "comando": comando,
        "gt": json.dumps(gt, ensure_ascii=False),
        "pred_raw": texto_generado,
        "pred_json": json.dumps(pred, ensure_ascii=False) if pred else "",
        "json_valido": json_valido,
        "parse_note": nota,
        "latencia_s": round(latencia_s, 3),
        **matches,
        "modo_prompting": modo,
        "transformers_version": transformers_version,
    }
    return {c: fila[c] for c in COLUMNAS_DETALLE}
```

- [ ] Correr y confirmar **verde** la parte pura: `pytest -q tests/test_run_sweep_2026.py`

### Tarea 2 — Inferencia y CLI

> Exención de TDD parcial: la inferencia real requiere descargar pesos y no es testeable unitariamente. Su corrección se verifica en el `Verify` con un modelo chico real y en el subtask 05. La lógica que **sí** es testeable ya quedó cubierta en la Tarea 1.

- [ ] Agregar a `src/run_sweep_2026.py` la inferencia, con los imports pesados dentro de la función:

```python
def evaluar_modelo(modelo: ModeloEvaluado2026, dataset: pd.DataFrame) -> list[dict]:
    import torch  # import local: los tests de lógica pura no necesitan torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    version = transformers.__version__
    print(f"=== {modelo.nombre} ({modelo.hf_repo_id}) | transformers {version} ===")

    tokenizer = AutoTokenizer.from_pretrained(
        modelo.hf_repo_id, trust_remote_code=modelo.trust_remote_code
    )
    red = AutoModelForCausalLM.from_pretrained(
        modelo.hf_repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=modelo.trust_remote_code,
    )
    red.eval()

    filas = []
    for idx, fila_ds in dataset.iterrows():
        comando = fila_ds["comando"]
        gt = fila_a_ground_truth(fila_ds)
        entrada, modo = construir_entrada(tokenizer, comando)

        inicio = time.perf_counter()
        with torch.no_grad():
            salida = red.generate(
                **entrada,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,     # greedy: RNF3, igual que el harness legacy
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencia_s = time.perf_counter() - inicio

        texto = tokenizer.decode(
            salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
        )
        fila = armar_fila(
            modelo=modelo, idx=idx, comando=comando, gt=gt,
            texto_generado=texto, latencia_s=latencia_s, modo=modo,
            transformers_version=version,
        )
        filas.append(fila)
        print(f"  [{idx:2d}] exact={fila['match_exact']} "
              f"lat={latencia_s:.2f}s modo={modo} {comando[:38]!r}")

    del red, tokenizer
    return filas


def main() -> int:
    parser = argparse.ArgumentParser(description="Barrido 2026, un modelo por corrida")
    parser.add_argument("--modelo", required=True,
                        help="nombre del modelo en el roster (ver src/models_2026.py)")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV del modelo ya esté completo")
    parser.add_argument("--listar", action="store_true",
                        help="listar los nombres del roster y salir")
    args = parser.parse_args()

    if args.listar:
        for m in MODELOS_2026:
            print(f"{m.nombre}\t{m.tier}\t{m.hf_repo_id}")
        return 0

    modelo = por_nombre(args.modelo)
    salida = ruta_detalle(modelo.nombre)
    if debe_saltear(salida, args.force):
        print(f"{modelo.nombre}: ya completo en {salida}, se saltea (usá --force)")
        return 0

    dataset = pd.read_csv(DATASET_PATH)
    if len(dataset) != N_COMANDOS_ESPERADO:
        raise ValueError(
            f"El dataset tiene {len(dataset)} filas, se esperaban {N_COMANDOS_ESPERADO}"
        )

    filas = evaluar_modelo(modelo, dataset)
    salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=COLUMNAS_DETALLE).to_csv(salida, index=False)
    exactas = sum(f["match_exact"] for f in filas)
    print(f"Listo: {len(filas)} filas en {salida} | exactas {exactas}/{len(filas)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Verificar que el CLI arranca sin descargar nada: `python src/run_sweep_2026.py --listar`
- [ ] Verificar que un nombre inválido falla claro:
  `python src/run_sweep_2026.py --modelo inexistente` → `ValueError` nombrando el roster

### Tarea 3 — commit

- [ ] `pytest -q`
- [ ] `git add src/run_sweep_2026.py tests/test_run_sweep_2026.py`
- [ ] `git commit -m "feat(2026): harness de barrido reanudable, un modelo por corrida"`

## Verify

```bash
# 1. Suite completa verde (la lógica pura corre sin torch instalado)
pytest -q

# 2. El roster se lista sin tocar la red
python src/run_sweep_2026.py --listar | wc -l    # -> 12

# 3. Las 15 primeras columnas coinciden exactamente con el CSV legacy
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from run_sweep_2026 import COLUMNAS_DETALLE
legacy = list(pd.read_csv('data/resultados_experimento_detalle.csv', nrows=1).columns)
assert COLUMNAS_DETALLE[:15] == legacy, (COLUMNAS_DETALLE[:15], legacy)
assert COLUMNAS_DETALLE[15:] == ['modo_prompting','transformers_version']
print('esquema de columnas OK')
"

# 4. Humo de inferencia real con el modelo más chico ya cacheado del roster
#    (requiere red la primera vez; ~5 min en CPU)
python src/run_sweep_2026.py --modelo "SmolLM2-360M-Instruct"
python -c "
import pandas as pd
df = pd.read_csv('data/2026/detalle/smollm2-360m-instruct.csv')
assert len(df) == 32, len(df)
assert df['modo_prompting'].unique().tolist() == ['chat_template']
assert df['idx'].tolist() == list(range(32))
print('barrido de humo OK, exactas:', int(df['match_exact'].sum()), '/32')
"

# 5. La reanudación saltea el modelo ya hecho
python src/run_sweep_2026.py --modelo "SmolLM2-360M-Instruct" | grep -q "se saltea" \
  && echo "reanudacion OK"

# 6. Archivos congelados intactos (F0)
git diff --exit-code main -- src/scoring.py src/run_evaluation.py \
  data/dataset_comandos_domotica.csv data/resultados_experimento_detalle.csv \
  && echo "F0 intacto"
```

## Acceptance criteria

- **Dado** `COLUMNAS_DETALLE`, **entonces** tiene exactamente las 17 columnas de F5 en ese orden, y sus 15 primeras son idénticas, una a una, a las del CSV legacy, de modo que `scoring.py` se reutiliza sin modificación.
- **Dado** un CSV de detalle con 32 filas y el esquema correcto, **cuando** se llama `debe_saltear(ruta, force=False)`, **entonces** devuelve `True`; **cuando** el CSV tiene menos de 32 filas, no existe, tiene otro esquema, o está corrupto, **entonces** devuelve `False`; **cuando** `force=True`, **entonces** siempre `False`.
- **Dado** un modelo y una respuesta JSON perfecta, **cuando** se llama `armar_fila`, **entonces** las claves salen en el orden de `COLUMNAS_DETALLE`, `json_valido` y `match_exact` son `True`, y `latencia_s` está redondeada a 3 decimales.
- **Dado** una respuesta que usa `"tele"` donde el ground truth dice `"tv"`, **cuando** se llama `armar_fila`, **entonces** `json_valido` es `True` pero `match_dispositivo` y `match_exact` son `False` — la etapa 1 penaliza el sinónimo, tal como exige el protocolo taxativo.
- **Dado** una respuesta no parseable, **entonces** `json_valido=False`, `pred_json=""`, `parse_note="no_parseable_como_json"` y todos los `match_*` en `False`.
- **Dado** un `modo` fuera de `{"chat_template", "raw_completion"}`, **entonces** `armar_fila` lanza `ValueError` mencionando `modo_prompting`.
- **Dado** `python src/run_sweep_2026.py --modelo "SmolLM2-360M-Instruct"`, **entonces** produce `data/2026/detalle/smollm2-360m-instruct.csv` con 32 filas, `idx` de 0 a 31 sin huecos, y una segunda invocación imprime que se saltea.
- **Dado** el módulo, **cuando** se importa desde los tests, **entonces** no requiere `torch` ni `transformers` (los imports pesados son locales a `evaluar_modelo`).
- **Dado** `git diff main`, **entonces** ningún archivo de F0 aparece modificado.
