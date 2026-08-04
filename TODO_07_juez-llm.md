---
id: 07
title: "Juez LLM: taxonomía de errores y categorías lingüísticas"
depends_on: [06]
files:
  - src/judge_2026.py
  - tests/test_judge_2026.py
---

## Spec

Implementar la etapa 2: el modelo elegido en `data/2026/juez_seleccionado.json` hace los **dos** trabajos de etiquetado que hoy el paper hace a mano.

- **RF8 / trabajo A** — clasificar cada respuesta con `match_exact == False` en un subconjunto no vacío de las 7 `ETIQUETAS_ERROR` (F4). Como el vocabulario incluye `sin_error_semantico` y `uso_de_sinonimos`, este mismo paso resuelve la adjudicación de equivalencia semántica que alimenta la métrica laxa (**RF10**).
- **RF9 / trabajo B** — clasificar cada uno de los 32 comandos en exactamente una de las 6 `CATEGORIAS_LINGUISTICAS` (F4), reemplazando el etiquetado manual de la Tabla 3. El dataset **no** se modifica.

Salida cerrada, nunca abierta: la respuesta del juez se parsea contra el vocabulario con `parsear_etiquetas` / `parsear_categoria`, con el **fallback congelado de F5** (un reintento con `max_new_tokens` duplicado; si falla, `juez_parse_ok=False` y etiqueta de respaldo determinista). Decodificación greedy (**RNF3**), así que reejecutar sobre el mismo detalle reproduce los mismos archivos byte a byte.

Este subtask escribe el **código**; el subtask 08 lo **ejecuta**. Todo lo testeable aquí se testea con un juez falso, sin descargar pesos.

**F12.3 — persistencia incremental.** `etiquetar_errores` y `etiquetar_categorias` ganan un parámetro
`persistir: Callable[[pd.DataFrame], None] | None = None`. Cuando no es `None` se lo invoca con el
DataFrame **acumulado hasta ese punto** (columnas ya en el orden de `COLUMNAS_ETIQUETAS` /
`COLUMNAS_CATEGORIAS`): en `etiquetar_errores`, al terminar cada **modelo** (agrupando `incorrectas`
por `modelo` en el orden del roster, para que "al terminar cada modelo" esté bien definido); en
`etiquetar_categorias`, al terminar cada **comando**. `persistir is None` reproduce exactamente el
comportamiento de antes de este contrato, así que los tests ya escritos en este archivo siguen
valiendo sin cambios. El `main()` pasa un persistidor que escribe el CSV **atómicamente**
(`<archivo>.tmp` + `rename`), de modo que el archivo en disco es siempre parseable. Dos flags nuevos
en la CLI: `--reanudar` (opt-in: salta los `(modelo, idx)` / `idx` ya presentes en el CSV en disco) y
el default sin flag, que recalcula todo — a propósito, para que el chequeo de determinismo (AC5 del
subtask 06 / C12 del plan de test) siga midiendo determinismo real y no la trivialidad de saltear
todo.

## Implementation plan

### Tarea 1 — Prompts del juez y parseo con fallback (TDD)

- [ ] Escribir el test que falla, `tests/test_judge_2026.py` (primera parte):

```python
"""Tests de la etapa 2 (juez LLM) con un juez falso: sin pesos, sin red."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_2026 import (  # noqa: E402
    COLUMNAS_CATEGORIAS,
    COLUMNAS_ETIQUETAS,
    etiqueta_de_respaldo,
    etiquetar_errores,
    etiquetar_categorias,
    prompt_categoria,
    prompt_error,
)
from taxonomia_2026 import CATEGORIAS_LINGUISTICAS, ETIQUETAS_ERROR  # noqa: E402


def _fila(idx=0, modelo="M", comando="Apagá la tele del living.",
          gt='{"intent": "apagar", "dispositivo": "tv", "ubicacion": "living", '
             '"valor": null, "unidad": null}',
          pred='{"intent": "apagar", "dispositivo": "tele", "ubicacion": "living", '
               '"valor": null, "unidad": null}',
          **overrides):
    base = {
        "modelo": modelo, "idx": idx, "comando": comando, "gt": gt,
        "pred_raw": pred, "pred_json": pred, "json_valido": True,
        "parse_note": "", "latencia_s": 1.0,
        "match_intent": True, "match_dispositivo": False, "match_ubicacion": True,
        "match_valor": True, "match_unidad": True, "match_exact": False,
        "modo_prompting": "chat_template", "transformers_version": "4.57.0",
    }
    base.update(overrides)
    return base


def test_el_prompt_de_error_lista_las_siete_etiquetas_y_nada_mas():
    p = prompt_error(_fila())
    for e in ETIQUETAS_ERROR:
        assert e in p
    assert "categorias cerradas" in p.lower() or "unicas etiquetas" in p.lower()


def test_el_prompt_de_error_incluye_comando_esperado_y_obtenido():
    p = prompt_error(_fila())
    assert "Apagá la tele del living." in p
    assert '"dispositivo": "tv"' in p      # esperado
    assert '"dispositivo": "tele"' in p    # obtenido


def test_el_prompt_de_categoria_lista_las_seis_categorias():
    p = prompt_categoria("Poné el aire de la cocina en 20 grados.")
    for c in CATEGORIAS_LINGUISTICAS:
        assert c in p
    assert "Poné el aire de la cocina en 20 grados." in p
    assert "exactamente una" in p.lower()


def test_respaldo_toma_el_primer_campo_que_no_coincide():
    assert etiqueta_de_respaldo(_fila(match_intent=False)) == "confusion_intencion"
    assert etiqueta_de_respaldo(_fila()) == "confusion_dispositivo"
    assert etiqueta_de_respaldo(
        _fila(match_dispositivo=True, match_ubicacion=False)
    ) == "confusion_ubicacion"


def test_respaldo_cuando_los_tres_campos_categoricos_coinciden():
    assert etiqueta_de_respaldo(
        _fila(match_dispositivo=True, match_valor=False)
    ) == "valor_numerico_incorrecto"
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_judge_2026.py`
- [ ] Implementar la primera parte de `src/judge_2026.py`:

```python
#!/usr/bin/env python3
"""
Etapa 2: el mejor modelo de la etapa 1 etiqueta automáticamente lo que el
paper original etiquetaba a mano.

Dos trabajos, un solo pipeline:
  A. cada respuesta incorrecta -> subconjunto no vacío de ETIQUETAS_ERROR
     (incluye sin_error_semantico / uso_de_sinonimos, que alimentan la
     métrica laxa);
  B. cada uno de los 32 comandos -> exactamente una CATEGORIA_LINGUISTICA.

La salida es cerrada por construcción: se parsea contra el vocabulario y, si
no parsea ni tras un reintento, se cae a una etiqueta de respaldo determinista
y se marca juez_parse_ok=False para poder reportar cuántas veces pasó.

Uso:
    python src/judge_2026.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from models_2026 import por_nombre
from taxonomia_2026 import (
    CATEGORIAS_DISPLAY,
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_ERROR,
    ETIQUETAS_ERROR_DISPLAY,
    SEP_ETIQUETAS,
    parsear_categoria,
    parsear_etiquetas,
)

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
PATH_CONSOLIDADO = DIR_2026 / "detalle_2026.csv"
PATH_JUEZ = DIR_2026 / "juez_seleccionado.json"
PATH_ETIQUETAS = DIR_2026 / "etiquetas_errores.csv"
PATH_CATEGORIAS = DIR_2026 / "categorias_comandos.csv"

COLUMNAS_ETIQUETAS = ["modelo", "idx", "etiquetas", "juez_raw", "juez_parse_ok"]
COLUMNAS_CATEGORIAS = ["idx", "comando", "categoria", "juez_raw", "juez_parse_ok"]

Persistidor = Callable[[pd.DataFrame], None]
"""F12.3: recibe el DataFrame acumulado hasta el punto en que se lo invoca."""

MAX_NEW_TOKENS_JUEZ = 48

_LISTA_ERRORES = "\n".join(
    f"- {e}: {ETIQUETAS_ERROR_DISPLAY[e]}" for e in ETIQUETAS_ERROR
)
_LISTA_CATEGORIAS = "\n".join(
    f"- {c}: {CATEGORIAS_DISPLAY[c]}" for c in CATEGORIAS_LINGUISTICAS
)


def prompt_error(fila: dict) -> str:
    """Prompt del trabajo A: clasificar una respuesta incorrecta."""
    return (
        "Sos un evaluador de un sistema de interpretación de comandos de "
        "domótica. Compará la salida ESPERADA con la OBTENIDA y clasificá la "
        "diferencia.\n\n"
        f"Comando: {fila['comando']}\n"
        f"Esperado: {fila['gt']}\n"
        f"Obtenido: {fila['pred_json'] or fila['pred_raw']}\n\n"
        "Estas son las unicas etiquetas admitidas (categorias cerradas):\n"
        f"{_LISTA_ERRORES}\n\n"
        "Respondé SOLO con una o más de esas etiquetas exactas, separadas por "
        f"'{SEP_ETIQUETAS}', sin explicación y sin ninguna palabra adicional. "
        "Usá sin_error_semantico si la salida significa lo mismo que la "
        "esperada, y uso_de_sinonimos si difiere solo por un sinónimo.\n"
        "Etiquetas:"
    )


def prompt_categoria(comando: str) -> str:
    """Prompt del trabajo B: clasificar lingüísticamente un comando."""
    return (
        "Clasificá el siguiente comando de domótica en español rioplatense "
        "según su tipo lingüístico.\n\n"
        f"Comando: {comando}\n\n"
        "Estas son las unicas categorias admitidas (categorias cerradas):\n"
        f"{_LISTA_CATEGORIAS}\n\n"
        "Respondé SOLO con exactamente una de esas categorias exactas, sin "
        "explicación y sin ninguna palabra adicional.\n"
        "Categoria:"
    )


def etiqueta_de_respaldo(fila: dict) -> str:
    """Respaldo determinista de F5 cuando el juez no produce salida parseable."""
    for campo, etiqueta in (
        ("match_intent", "confusion_intencion"),
        ("match_dispositivo", "confusion_dispositivo"),
        ("match_ubicacion", "confusion_ubicacion"),
    ):
        if not fila[campo]:
            return etiqueta
    return "valor_numerico_incorrecto"
```

- [ ] Correr y confirmar **verde** la primera parte: `pytest -q tests/test_judge_2026.py`

### Tarea 2 — Bucle de etiquetado con reintento y fallback (TDD)

- [ ] Agregar a `tests/test_judge_2026.py`:

```python
class JuezFalso:
    """Devuelve respuestas programadas; registra cuántas veces se lo llamó."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.llamadas.append((prompt, max_new_tokens))
        return self.respuestas.pop(0) if self.respuestas else ""


def test_etiquetar_errores_solo_procesa_las_incorrectas():
    df = pd.DataFrame([
        _fila(idx=0, match_exact=True, match_dispositivo=True),
        _fila(idx=1),
        _fila(idx=2, match_exact=True, match_dispositivo=True),
    ])
    juez = JuezFalso(["uso_de_sinonimos"])
    out = etiquetar_errores(df, juez)
    assert list(out.columns) == COLUMNAS_ETIQUETAS
    assert len(out) == 1 and out.iloc[0]["idx"] == 1
    assert len(juez.llamadas) == 1


def test_etiquetar_errores_normaliza_y_marca_parse_ok():
    df = pd.DataFrame([_fila(idx=1)])
    out = etiquetar_errores(df, JuezFalso(["  Uso_De_Sinonimos ;confusion_dispositivo "]))
    assert out.iloc[0]["etiquetas"] == "uso_de_sinonimos;confusion_dispositivo"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_reintenta_una_vez_con_el_doble_de_tokens():
    df = pd.DataFrame([_fila(idx=1)])
    juez = JuezFalso(["blah blah no es una etiqueta", "sin_error_semantico"])
    out = etiquetar_errores(df, juez)
    assert len(juez.llamadas) == 2
    assert juez.llamadas[1][1] == 2 * juez.llamadas[0][1]
    assert out.iloc[0]["etiquetas"] == "sin_error_semantico"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_cae_al_respaldo_tras_dos_fallos_y_marca_parse_ok_false():
    df = pd.DataFrame([_fila(idx=1)])
    out = etiquetar_errores(df, JuezFalso(["ni idea", "tampoco"]))
    assert out.iloc[0]["etiquetas"] == "confusion_dispositivo"
    assert bool(out.iloc[0]["juez_parse_ok"]) is False
    assert out.iloc[0]["juez_raw"] == "tampoco"


def test_toda_fila_incorrecta_recibe_al_menos_una_etiqueta_valida():
    df = pd.DataFrame([_fila(idx=i) for i in range(5)])
    out = etiquetar_errores(df, JuezFalso(["basura"] * 20))
    assert len(out) == 5
    for etiquetas in out["etiquetas"]:
        partes = etiquetas.split(";")
        assert partes and all(p in ETIQUETAS_ERROR for p in partes)


def test_etiquetar_categorias_cubre_los_comandos_una_sola_vez():
    df = pd.DataFrame([
        _fila(idx=0, modelo="A", comando="Prendé la luz."),
        _fila(idx=1, modelo="A", comando="Poné el aire en 20."),
        _fila(idx=0, modelo="B", comando="Prendé la luz."),   # mismo comando, otro modelo
    ])
    juez = JuezFalso(["encendido_apagado_simple", "ajuste_con_valor_numerico"])
    out = etiquetar_categorias(df, juez)
    assert list(out.columns) == COLUMNAS_CATEGORIAS
    assert out["idx"].tolist() == [0, 1]          # una vez por comando, no por corrida
    assert len(juez.llamadas) == 2


def test_categoria_invalida_cae_al_respaldo():
    df = pd.DataFrame([_fila(idx=0, comando="Prendé la luz.")])
    out = etiquetar_categorias(df, JuezFalso(["inventada", "tampoco"]))
    assert out.iloc[0]["categoria"] == "encendido_apagado_simple"
    assert bool(out.iloc[0]["juez_parse_ok"]) is False


def test_categoria_con_dos_valores_no_parsea():
    df = pd.DataFrame([_fila(idx=0, comando="Prendé la luz.")])
    out = etiquetar_categorias(
        df, JuezFalso(["consulta_de_estado;multiples_dispositivos", "consulta_de_estado"])
    )
    assert out.iloc[0]["categoria"] == "consulta_de_estado"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_etiquetar_errores_invoca_persistir_una_vez_por_modelo_con_acumulado_creciente():
    """F12.3: una llamada por modelo, cada una con más filas que la anterior."""
    df = pd.DataFrame([
        _fila(idx=0, modelo="A"),
        _fila(idx=1, modelo="A"),
        _fila(idx=0, modelo="B"),
    ])
    juez = JuezFalso(["confusion_dispositivo"] * 3)
    llamados = []
    etiquetar_errores(df, juez, persistir=lambda acc: llamados.append(acc.copy()))
    assert len(llamados) == 2
    assert len(llamados[0]) == 2 and llamados[0]["modelo"].tolist() == ["A", "A"]
    assert len(llamados[1]) == 3
    assert all(list(acc.columns) == COLUMNAS_ETIQUETAS for acc in llamados)


def test_etiquetar_categorias_invoca_persistir_una_vez_por_comando():
    df = pd.DataFrame([
        _fila(idx=0, comando="Prendé la luz."),
        _fila(idx=1, comando="Poné el aire en 20."),
    ])
    juez = JuezFalso(["encendido_apagado_simple", "ajuste_con_valor_numerico"])
    llamados = []
    etiquetar_categorias(df, juez, persistir=lambda acc: llamados.append(acc.copy()))
    assert len(llamados) == 2
    assert len(llamados[0]) == 1 and len(llamados[1]) == 2
    assert all(list(acc.columns) == COLUMNAS_CATEGORIAS for acc in llamados)


def test_persistir_none_reproduce_el_resultado_de_antes_del_contrato():
    df = pd.DataFrame([_fila(idx=0, modelo="A"), _fila(idx=1, modelo="B")])
    sin_callback = etiquetar_errores(df, JuezFalso(["confusion_dispositivo"] * 2))
    con_persistir_none = etiquetar_errores(df, JuezFalso(["confusion_dispositivo"] * 2), persistir=None)
    pd.testing.assert_frame_equal(sin_callback, con_persistir_none)
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_judge_2026.py`
- [ ] Implementar el bucle genérico y los dos etiquetadores:

```python
def _consultar(juez, prompt: str, parsear, respaldo):
    """Un intento, un reintento con el doble de tokens, y si no, respaldo (F5)."""
    crudo = ""
    for tokens in (MAX_NEW_TOKENS_JUEZ, 2 * MAX_NEW_TOKENS_JUEZ):
        crudo = juez(prompt, tokens)
        try:
            return parsear(crudo), crudo, True
        except ValueError:
            continue
    return respaldo(), crudo, False


def etiquetar_errores(df_detalle: pd.DataFrame, juez,
                      persistir: Persistidor | None = None) -> pd.DataFrame:
    """Trabajo A: una fila por respuesta incorrecta (RF8).

    F12.3: si se pasa `persistir`, se lo invoca con el acumulado al terminar
    cada **modelo** -- agrupando `incorrectas` por `modelo` en el orden en
    que aparece en `df_detalle` (orden de roster, el que deja el subtask 06),
    para que "al terminar cada modelo" quede bien definido. `persistir is
    None` reproduce exactamente el comportamiento anterior a este contrato.
    """
    filas = []
    incorrectas = df_detalle[~df_detalle["match_exact"].astype(bool)]
    for _, grupo in incorrectas.groupby("modelo", sort=False):
        for _, fila in grupo.iterrows():
            d = fila.to_dict()
            etiquetas, crudo, ok = _consultar(
                juez,
                prompt_error(d),
                parsear_etiquetas,
                lambda d=d: [etiqueta_de_respaldo(d)],
            )
            filas.append({
                "modelo": d["modelo"],
                "idx": int(d["idx"]),
                "etiquetas": SEP_ETIQUETAS.join(etiquetas),
                "juez_raw": crudo,
                "juez_parse_ok": ok,
            })
        if persistir is not None:
            persistir(pd.DataFrame(filas, columns=COLUMNAS_ETIQUETAS))
    return pd.DataFrame(filas, columns=COLUMNAS_ETIQUETAS)


def etiquetar_categorias(df_detalle: pd.DataFrame, juez,
                         persistir: Persistidor | None = None) -> pd.DataFrame:
    """Trabajo B: una fila por comando del dataset, no por corrida (RF9).

    F12.3: si se pasa `persistir`, se lo invoca con el acumulado al terminar
    cada **comando**. `persistir is None` reproduce el comportamiento
    anterior a este contrato.
    """
    comandos = (
        df_detalle[["idx", "comando"]]
        .drop_duplicates(subset="idx")
        .sort_values("idx")
    )
    filas = []
    for _, fila in comandos.iterrows():
        categoria, crudo, ok = _consultar(
            juez,
            prompt_categoria(fila["comando"]),
            parsear_categoria,
            lambda: CATEGORIAS_LINGUISTICAS[0],
        )
        filas.append({
            "idx": int(fila["idx"]),
            "comando": fila["comando"],
            "categoria": categoria,
            "juez_raw": crudo,
            "juez_parse_ok": ok,
        })
        if persistir is not None:
            persistir(pd.DataFrame(filas, columns=COLUMNAS_CATEGORIAS))
    return pd.DataFrame(filas, columns=COLUMNAS_CATEGORIAS)
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_judge_2026.py`

### Tarea 3 — Escritor atómico y filtrado de `--reanudar` (F12.3, TDD)

- [ ] Agregar a `tests/test_judge_2026.py`:

```python
from judge_2026 import (  # noqa: E402
    construir_persistidor_atomico,
    filtrar_pendientes_categorias,
    filtrar_pendientes_errores,
)


def test_escritor_atomico_no_deja_tmp_y_el_csv_parsea_tras_cada_llamada(tmp_path):
    destino = tmp_path / "salida.csv"
    escribir = construir_persistidor_atomico(destino)
    for n in (1, 2, 3):
        escribir(pd.DataFrame({"idx": range(n), "valor": range(n)}))
        assert destino.exists()
        assert not (tmp_path / "salida.csv.tmp").exists()
        assert len(pd.read_csv(destino)) == n


def test_escritor_atomico_con_base_antepone_lo_ya_persistido(tmp_path):
    destino = tmp_path / "salida.csv"
    base = pd.DataFrame({"idx": [0, 1], "valor": [10, 11]})
    escribir = construir_persistidor_atomico(destino, base=base)
    escribir(pd.DataFrame({"idx": [2], "valor": [12]}))
    assert pd.read_csv(destino)["idx"].tolist() == [0, 1, 2]


def test_reanudar_salta_los_pares_modelo_idx_ya_presentes():
    df = pd.DataFrame([
        _fila(idx=0, modelo="A"), _fila(idx=1, modelo="A"), _fila(idx=0, modelo="B"),
    ])
    incorrectas = df[~df["match_exact"].astype(bool)]
    pendientes = filtrar_pendientes_errores(incorrectas, ya_hechos={("A", 0)})
    assert sorted(map(tuple, pendientes[["modelo", "idx"]].values)) == [("A", 1), ("B", 0)]


def test_reanudar_categorias_salta_los_idx_ya_presentes():
    df = pd.DataFrame([_fila(idx=i, comando=f"c{i}") for i in range(4)])
    comandos = df[["idx", "comando"]].drop_duplicates(subset="idx").sort_values("idx")
    pendientes = filtrar_pendientes_categorias(comandos, ya_hechos={0, 2})
    assert pendientes["idx"].tolist() == [1, 3]
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_judge_2026.py`
- [ ] Implementar en `src/judge_2026.py`:

```python
def construir_persistidor_atomico(destino: Path, base: pd.DataFrame | None = None) -> Persistidor:
    """F12.3: escribe `<destino>.tmp` y hace `rename`, para que el archivo en
    disco sea siempre parseable. Si `base` no es `None` (modo `--reanudar`),
    cada llamada persiste `base` concatenado con lo nuevo del lote."""
    tmp = destino.with_suffix(destino.suffix + ".tmp")

    def persistir(acumulado: pd.DataFrame) -> None:
        completo = (
            pd.concat([base, acumulado], ignore_index=True) if base is not None else acumulado
        )
        completo.to_csv(tmp, index=False)
        tmp.replace(destino)

    return persistir


def filtrar_pendientes_errores(incorrectas: pd.DataFrame, ya_hechos: set[tuple]) -> pd.DataFrame:
    """`--reanudar`: descarta pares (modelo, idx) ya presentes en el CSV en disco."""
    hecho = incorrectas.apply(lambda f: (f["modelo"], int(f["idx"])) in ya_hechos, axis=1)
    return incorrectas[~hecho]


def filtrar_pendientes_categorias(comandos: pd.DataFrame, ya_hechos: set[int]) -> pd.DataFrame:
    """`--reanudar`: descarta los `idx` de comando ya presentes en el CSV en disco."""
    return comandos[~comandos["idx"].isin(ya_hechos)]
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_judge_2026.py`

### Tarea 4 — Juez real y CLI

> Exención de TDD: la carga del modelo real no es testeable unitariamente; su comportamiento se ejerce en el subtask 08. La lógica que lo rodea ya está cubierta.

- [ ] Agregar la fábrica del juez real y el `main`:

```python
def construir_juez_real(hf_repo_id: str, trust_remote_code: bool):
    """Devuelve un callable (prompt, max_new_tokens) -> texto, greedy (RNF3)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prompt_2026 import detectar_modo

    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo_id, trust_remote_code=trust_remote_code
    )
    red = AutoModelForCausalLM.from_pretrained(
        hf_repo_id, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    red.eval()
    modo = detectar_modo(tokenizer)

    def juez(prompt: str, max_new_tokens: int) -> str:
        if modo == "chat_template":
            entrada = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        else:
            entrada = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            salida = red.generate(
                **entrada, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    return juez


def main() -> int:
    parser = argparse.ArgumentParser(description="Etapa 2: etiquetado con juez LLM")
    parser.add_argument("--detalle", default=str(PATH_CONSOLIDADO))
    parser.add_argument("--juez-json", default=str(PATH_JUEZ))
    parser.add_argument(
        "--reanudar", action="store_true",
        help="Opt-in (F12.3): salta (modelo, idx) / idx ya presentes en los CSV en "
             "disco. El default, sin este flag, recalcula todo, para que el chequeo "
             "de determinismo (subtask 06 AC5 / TEST_PLAN C12) siga midiendo "
             "determinismo real y no la trivialidad de saltear todo.",
    )
    args = parser.parse_args()

    info = json.loads(Path(args.juez_json).read_text(encoding="utf-8"))
    modelo = por_nombre(info["modelo"])
    print(f"Juez: {modelo.nombre} ({modelo.hf_repo_id}) "
          f"| exact_match etapa 1 = {info['exact_match_pct']}%")

    df = pd.read_csv(args.detalle)
    juez = construir_juez_real(modelo.hf_repo_id, modelo.trust_remote_code)

    PATH_ETIQUETAS.parent.mkdir(parents=True, exist_ok=True)
    base_etiquetas = pd.read_csv(PATH_ETIQUETAS) if args.reanudar and PATH_ETIQUETAS.exists() else None
    base_categorias = pd.read_csv(PATH_CATEGORIAS) if args.reanudar and PATH_CATEGORIAS.exists() else None

    df_para_errores = df
    if base_etiquetas is not None:
        ya_hechos = set(map(tuple, base_etiquetas[["modelo", "idx"]].values))
        incorrectas = df[~df["match_exact"].astype(bool)]
        pendientes = filtrar_pendientes_errores(incorrectas, ya_hechos)
        correctas = df[df["match_exact"].astype(bool)]
        df_para_errores = pd.concat([correctas, pendientes], ignore_index=True)

    df_para_categorias = df
    if base_categorias is not None:
        comandos = df[["idx", "comando"]].drop_duplicates(subset="idx").sort_values("idx")
        pendientes_c = filtrar_pendientes_categorias(comandos, set(base_categorias["idx"]))
        df_para_categorias = df[df["idx"].isin(pendientes_c["idx"])]

    df_etiquetas_nuevas = etiquetar_errores(
        df_para_errores, juez,
        persistir=construir_persistidor_atomico(PATH_ETIQUETAS, base=base_etiquetas),
    )
    df_categorias_nuevas = etiquetar_categorias(
        df_para_categorias, juez,
        persistir=construir_persistidor_atomico(PATH_CATEGORIAS, base=base_categorias),
    )

    df_etiquetas = (
        pd.concat([base_etiquetas, df_etiquetas_nuevas], ignore_index=True)
        if base_etiquetas is not None else df_etiquetas_nuevas
    )
    df_categorias = (
        pd.concat([base_categorias, df_categorias_nuevas], ignore_index=True)
        if base_categorias is not None else df_categorias_nuevas
    )

    fallidas = int((~df_etiquetas["juez_parse_ok"]).sum())
    fallidas_c = int((~df_categorias["juez_parse_ok"]).sum())
    print(f"Etiquetas: {len(df_etiquetas)} filas ({fallidas} sin parsear) -> {PATH_ETIQUETAS}")
    print(f"Categorías: {len(df_categorias)} filas ({fallidas_c} sin parsear) -> {PATH_CATEGORIAS}")
    print(df_categorias["categoria"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Confirmar que el módulo se importa sin `torch`:
  `python -c "import sys; sys.path.insert(0,'src'); import judge_2026; assert 'torch' not in sys.modules; print('ok')"`

### Tarea 5 — commit

- [ ] `pytest -q`
- [ ] `git add src/judge_2026.py tests/test_judge_2026.py`
- [ ] `git commit -m "feat(2026): juez LLM con salida cerrada para taxonomia de errores y categorias"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Los prompts son cerrados: listan todo el vocabulario y nada más
python -c "
import sys; sys.path.insert(0,'src')
from judge_2026 import prompt_error, prompt_categoria
from taxonomia_2026 import ETIQUETAS_ERROR, CATEGORIAS_LINGUISTICAS
f = {'comando':'Apagá la tele.','gt':'{\"dispositivo\": \"tv\"}',
     'pred_json':'{\"dispositivo\": \"tele\"}','pred_raw':''}
pe, pc = prompt_error(f), prompt_categoria('Prendé la luz.')
assert all(e in pe for e in ETIQUETAS_ERROR)
assert all(c in pc for c in CATEGORIAS_LINGUISTICAS)
assert 'SOLO' in pe and 'SOLO' in pc
print('prompts cerrados OK')
"

# 3. Nunca queda una fila incorrecta sin etiqueta, ni con el juez más inútil posible
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from judge_2026 import etiquetar_errores, etiquetar_categorias
from taxonomia_2026 import ETIQUETAS_ERROR, CATEGORIAS_LINGUISTICAS
from run_sweep_2026 import COLUMNAS_DETALLE
base = dict(zip(COLUMNAS_DETALLE, [None]*len(COLUMNAS_DETALLE)))
filas = []
for i in range(10):
    f = dict(base, modelo='M', idx=i, comando=f'c{i}', gt='{}', pred_json='{}',
             pred_raw='', match_intent=True, match_dispositivo=False,
             match_ubicacion=True, match_valor=True, match_unidad=True,
             match_exact=False)
    filas.append(f)
df = pd.DataFrame(filas, columns=COLUMNAS_DETALLE)
inutil = lambda p, t: 'no se'
e = etiquetar_errores(df, inutil); c = etiquetar_categorias(df, inutil)
assert len(e) == 10 and all(x in ETIQUETAS_ERROR for v in e['etiquetas'] for x in v.split(';'))
assert len(c) == 10 and c['categoria'].isin(CATEGORIAS_LINGUISTICAS).all()
assert not e['juez_parse_ok'].any()
print('fallback total OK')
"

# 4. El módulo no arrastra torch al importarse
python -c "
import sys; sys.path.insert(0,'src'); import judge_2026
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('imports livianos OK')
"

# 5. El dataset original no se tocó
git diff --exit-code main -- data/dataset_comandos_domotica.csv && echo "dataset intacto"
```

## Acceptance criteria

- **Dado** `prompt_error`, **entonces** el prompt incluye el comando, el esperado y el obtenido, lista las **7** `ETIQUETAS_ERROR` con su nombre legible, declara que son las únicas admitidas, y pide responder solo con etiquetas separadas por `;`.
- **Dado** `prompt_categoria`, **entonces** lista las **6** `CATEGORIAS_LINGUISTICAS`, incluye el comando y pide exactamente una.
- **Dado** un detalle con filas correctas e incorrectas, **cuando** se corre `etiquetar_errores`, **entonces** produce una fila por cada `match_exact == False` y **ninguna** por las correctas, con las columnas `COLUMNAS_ETIQUETAS` en orden.
- **Dado** una salida del juez con mayúsculas, espacios y duplicados, **entonces** se normaliza a etiquetas válidas en minúsculas, sin duplicados, preservando el orden.
- **Dado** una primera salida no parseable, **entonces** se reintenta **exactamente una** vez con el doble de `max_new_tokens`; si el reintento parsea, `juez_parse_ok` es `True`.
- **Dado** dos salidas no parseables seguidas, **entonces** se usa la etiqueta de respaldo determinista (primer `confusion_*` cuyo campo no coincide, en orden `intent, dispositivo, ubicacion`; si los tres coinciden, `valor_numerico_incorrecto`), `juez_parse_ok` es `False`, y `juez_raw` guarda la última salida cruda.
- **Dado** un juez que nunca produce salida válida, **entonces** **toda** fila incorrecta igualmente termina con al menos una etiqueta perteneciente a `ETIQUETAS_ERROR`: el pipeline no deja huecos.
- **Dado** un detalle con los 12 modelos del roster activo, **cuando** se corre `etiquetar_categorias`, **entonces** produce una fila por **comando** (`idx` único), no por corrida, ordenadas por `idx`, y consulta al juez una sola vez por comando.
- **Dado** una respuesta del juez con dos categorías, **entonces** no parsea y se reintenta; si el reintento da una sola válida, se acepta con `juez_parse_ok=True`.
- **Dado** `src/judge_2026.py`, **cuando** se lo importa, **entonces** no carga `torch` ni `transformers` (viven dentro de `construir_juez_real`), y el juez real usa `do_sample=False`, `temperature=None`, `top_p=None`.
- **Dado** un `persistir` no nulo, **cuando** se corre `etiquetar_errores`, **entonces** se lo invoca exactamente una vez por modelo (agrupado en el orden en que aparece en `df_detalle`) con el DataFrame acumulado hasta ese punto, de tamaño monótonamente creciente y con columnas siempre iguales a `COLUMNAS_ETIQUETAS`; **dado** `etiquetar_categorias`, se lo invoca una vez por comando con columnas siempre `COLUMNAS_CATEGORIAS`.
- **Dado** `persistir=None` en cualquiera de los dos etiquetadores, **entonces** el resultado es byte-idéntico (mismas filas, mismo orden, mismas columnas) al de antes de introducir F12.3.
- **Dado** `construir_persistidor_atomico`, **entonces** cada llamada deja el archivo destino parseable por `pandas.read_csv` y **nunca** deja un `<archivo>.tmp` residual; con `base` no nulo, el CSV resultante antepone `base` a lo nuevo del lote.
- **Dado** `--reanudar`, **entonces** `filtrar_pendientes_errores` descarta los pares `(modelo, idx)` ya presentes en el CSV en disco y `filtrar_pendientes_categorias` descarta los `idx` ya presentes; **dado** el default (sin `--reanudar`), **entonces** se recalcula todo el detalle sin filtrar nada, para que el chequeo de determinismo (subtask 06 AC5 / TEST_PLAN C12) siga midiendo determinismo real y no la trivialidad de saltear todo.
- **Dado** el commit, **entonces** `pytest -q` pasa y `data/dataset_comandos_domotica.csv` sigue intacto.
