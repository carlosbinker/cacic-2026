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

from metrics_2026 import _como_bool
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
    incorrectas = df_detalle[~_como_bool(df_detalle["match_exact"])]
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
        incorrectas = df[~_como_bool(df["match_exact"])]
        pendientes = filtrar_pendientes_errores(incorrectas, ya_hechos)
        correctas = df[_como_bool(df["match_exact"])]
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
