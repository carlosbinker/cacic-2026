#!/usr/bin/env python3
"""
Harness de barrido 2026: evalúa UN modelo del roster sobre los 32 comandos y
escribe data/2026/detalle/<slug>.csv con las 17 columnas congeladas en F5.

Un modelo por invocación, a propósito: cada uno corre en su propia imagen
Docker (subtask 04) con su propia versión de transformers, y el barrido
completo debe ser reanudable --- son ~4-8 h de CPU y una interrupción no
puede obligar a rehacer los modelos ya terminados (RF3).

La lógica pura (armado de fila, decisión de saltear, chequeo de credencial)
vive separada de la inferencia y sin imports de torch/transformers a nivel de
módulo, para poder testearla sin descargar pesos.

Uso:
    python src/run_sweep_2026.py --modelo "Qwen3.5-0.8B" [--force]
    python src/run_sweep_2026.py --listar
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# Tipos de transformers/torch, que no se importan a nivel de módulo para que
# la lógica pura siga siendo testeable sin esas dependencias.
Tokenizer = Any
RedCausal = Any


# --------------------------------------------------------------------------
# Lógica pura: reanudación, credenciales y armado de fila
# --------------------------------------------------------------------------

def ruta_detalle(nombre_modelo: str) -> Path:
    """CSV de detalle que le corresponde a un modelo del roster."""
    return DIR_DETALLE / f"{slug(nombre_modelo)}.csv"


def _esta_completo(df: pd.DataFrame) -> bool:
    """True si el DataFrame es un barrido entero y bien formado de un modelo.

    No alcanza con contar filas: un barrido cortado a mitad de escritura puede
    dejar un CSV con el largo correcto pero con índices repetidos o faltantes,
    y darlo por bueno costaría un modelo entero de resultados falsos.
    """
    if list(df.columns) != COLUMNAS_DETALLE or len(df) != N_COMANDOS_ESPERADO:
        return False
    try:
        indices = sorted(int(i) for i in df["idx"])
    except (TypeError, ValueError):
        return False
    return indices == list(range(N_COMANDOS_ESPERADO))


def debe_saltear(ruta: Path, force: bool) -> bool:
    """True si ese modelo ya tiene un CSV completo y bien formado (RF3)."""
    if force or not ruta.exists():
        return False
    try:
        df = pd.read_csv(ruta)
    except Exception:
        return False
    return _esta_completo(df)


def verificar_credencial_gated(modelo: ModeloEvaluado2026) -> None:
    """Falla temprano (F11): antes de tocar la red, si el modelo es `gated` y
    falta `$HF_TOKEN`, aborta en vez de fallar a mitad de una descarga. El
    mensaje nunca imprime el valor del token."""
    if modelo.gated and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            f"{modelo.nombre} es un modelo gated: requiere licencia aceptada y "
            f"un token de Hugging Face para descargarse. Definí HF_TOKEN en el "
            f"archivo .env de la raíz del repo (ver docker/README.md)."
        )


def armar_fila(*, modelo: ModeloEvaluado2026, idx: int, comando: str,
               gt: dict[str, Any], texto_generado: str, latencia_s: float,
               modo: str, transformers_version: str) -> dict[str, Any]:
    """Arma una fila del CSV de detalle con las 17 columnas de F5, en orden.

    Reutiliza `scoring.py` tal cual (F0): la etapa 1 es coincidencia textual
    exacta campo a campo (RF6), así que un sinónimo cuenta como error.
    """
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


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextoInferencia:
    """Lo que el bucle de generación necesita y no cambia entre comandos."""

    modelo: ModeloEvaluado2026
    tokenizer: Tokenizer
    red: RedCausal
    transformers_version: str


def _credencial_hf(modelo: ModeloEvaluado2026) -> dict[str, str]:
    """F11/RF17: solo los modelos gated reciben el token, y solo como
    parámetro `token=` de `from_pretrained`. Nunca se hace login interactivo
    de huggingface_hub, nunca se escribe en disco, nunca se imprime."""
    return {"token": os.environ["HF_TOKEN"]} if modelo.gated else {}


def _abrir_contexto(modelo: ModeloEvaluado2026) -> ContextoInferencia:
    """Descarga/carga pesos en CPU y deja el modelo listo para generar."""
    import torch  # import local: los tests de lógica pura no necesitan torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    version = transformers.__version__
    print(f"=== {modelo.nombre} ({modelo.hf_repo_id}) | transformers {version} ===")

    credencial = _credencial_hf(modelo)
    tokenizer = AutoTokenizer.from_pretrained(
        modelo.hf_repo_id,
        trust_remote_code=modelo.trust_remote_code,
        **credencial,
    )
    red = AutoModelForCausalLM.from_pretrained(
        modelo.hf_repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=modelo.trust_remote_code,
        **credencial,
    )
    red.eval()
    return ContextoInferencia(
        modelo=modelo, tokenizer=tokenizer, red=red, transformers_version=version
    )


def _generar_respuesta(ctx: ContextoInferencia,
                       entrada: dict[str, Any]) -> tuple[str, float]:
    """Genera la continuación greedy y devuelve (texto_nuevo, latencia_s).

    Decodificación idéntica a la del harness legacy (RNF3): sin muestreo, sin
    temperatura y sin top_p, para que la tabla de latencia y las métricas sigan
    siendo comparables con las publicadas.
    """
    import torch  # import local, ver _abrir_contexto

    inicio = time.perf_counter()
    with torch.no_grad():
        salida = ctx.red.generate(
            **entrada,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,     # greedy: RNF3, igual que el harness legacy
            temperature=None,
            top_p=None,
            pad_token_id=ctx.tokenizer.eos_token_id,
        )
    latencia_s = time.perf_counter() - inicio

    texto = ctx.tokenizer.decode(
        salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return texto, latencia_s


def _evaluar_comando(ctx: ContextoInferencia, idx: int,
                     fila_ds: pd.Series) -> dict[str, Any]:
    """Corre un comando del dataset y devuelve su fila de detalle."""
    comando = fila_ds["comando"]
    entrada, modo = construir_entrada(ctx.tokenizer, comando)
    texto, latencia_s = _generar_respuesta(ctx, entrada)
    return armar_fila(
        modelo=ctx.modelo,
        idx=idx,
        comando=comando,
        gt=fila_a_ground_truth(fila_ds),
        texto_generado=texto,
        latencia_s=latencia_s,
        modo=modo,
        transformers_version=ctx.transformers_version,
    )


def evaluar_modelo(modelo: ModeloEvaluado2026,
                   dataset: pd.DataFrame) -> list[dict[str, Any]]:
    """Evalúa el modelo sobre todo el dataset y devuelve las 32 filas."""
    verificar_credencial_gated(modelo)  # F11: antes de descargar/cargar pesos
    ctx = _abrir_contexto(modelo)

    filas: list[dict[str, Any]] = []
    for idx, fila_ds in dataset.iterrows():
        fila = _evaluar_comando(ctx, int(idx), fila_ds)
        filas.append(fila)
        print(f"  [{fila['idx']:2d}] exact={fila['match_exact']} "
              f"lat={fila['latencia_s']:.2f}s modo={fila['modo_prompting']} "
              f"{fila['comando'][:38]!r}")

    del ctx
    return filas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cargar_dataset() -> pd.DataFrame:
    """Lee el dataset congelado de 32 comandos y verifica su tamaño."""
    dataset = pd.read_csv(DATASET_PATH)
    if len(dataset) != N_COMANDOS_ESPERADO:
        raise ValueError(
            f"El dataset {DATASET_PATH} tiene {len(dataset)} filas, "
            f"se esperaban {N_COMANDOS_ESPERADO}"
        )
    return dataset


def escribir_detalle(filas: list[dict[str, Any]], salida: Path) -> None:
    """Escribe el CSV de detalle del modelo con las 17 columnas de F5."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=COLUMNAS_DETALLE).to_csv(salida, index=False)
    exactas = sum(f["match_exact"] for f in filas)
    print(f"Listo: {len(filas)} filas en {salida} | exactas {exactas}/{len(filas)}")


def listar_roster() -> None:
    """Imprime el roster, una línea por modelo, sin tocar la red."""
    for modelo in MODELOS_2026:
        print(f"{modelo.nombre}\t{modelo.tier}\t{modelo.hf_repo_id}")


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido 2026, un modelo por corrida"
    )
    parser.add_argument("--modelo",
                        help="nombre del modelo en el roster (ver src/models_2026.py)")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV del modelo ya esté completo")
    parser.add_argument("--listar", action="store_true",
                        help="listar los nombres del roster y salir")
    args = parser.parse_args()
    if not args.listar and not args.modelo:
        parser.error("falta --modelo <nombre> (o usá --listar para ver el roster)")
    return args


def main() -> int:
    args = _parsear_argumentos()
    if args.listar:
        listar_roster()
        return 0

    modelo = por_nombre(args.modelo)
    salida = ruta_detalle(modelo.nombre)
    if debe_saltear(salida, args.force):
        print(f"{modelo.nombre}: ya completo en {salida}, se saltea (usá --force)")
        return 0

    escribir_detalle(evaluar_modelo(modelo, cargar_dataset()), salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
