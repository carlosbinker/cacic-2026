#!/usr/bin/env python3
"""
Harness de evaluación de los cuatro SLMs sobre el dataset de comandos de
domótica en español rioplatense.

RECONSTRUCCIÓN DOCUMENTADA -- no es el script original.
=========================================================
El script que efectivamente generó data/resultados_experimento_detalle.csv
y data/resultados_experimento_resumen.json corrió en un entorno de sesión
efímero (un contenedor cloud que se descarta al terminar la sesión) y
nunca se guardó como archivo. Solo sobrevivieron sus salidas.

Este archivo reimplementa esa evaluación siguiendo al pie de la letra la
metodología documentada en el paper (Sección 3: modelos, dataset, prompt,
decodificación, métricas). Fue validado estructuralmente contra el CSV de
detalle real -- ver tests/test_metricas.py y
src/validar_contra_resultados_originales.py -- pero al ser cuatro
descargas de modelo + inferencia real, sus resultados numéricos pueden
diferir levemente de los originales según versión de librerías, hardware
y build de PyTorch. No se ejecutó de punta a punta en esta sesión (correr
los cuatro modelos completos no era necesario para validar la fidelidad
metodológica del pipeline; podés correrlo vos con
`python src/run_evaluation.py`).

Uso:
    python src/run_evaluation.py [--salida data/resultados_reproducidos.csv]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODELOS
from prompt import SYSTEM_PROMPT_RECONSTRUIDO, construir_prompt_usuario
from scoring import comparar_campos, extraer_json, fila_a_ground_truth

RAIZ = Path(__file__).resolve().parent.parent
DATASET_PATH = RAIZ / "data" / "dataset_comandos_domotica.csv"


def cargar_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def evaluar_modelo(modelo_info, dataset: pd.DataFrame) -> list[dict]:
    print(f"\n=== Cargando {modelo_info.nombre} ({modelo_info.hf_repo_id}) ===")
    tokenizer = AutoTokenizer.from_pretrained(modelo_info.hf_repo_id)
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_info.hf_repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    modelo.eval()

    filas_resultado = []
    for idx, fila in dataset.iterrows():
        comando = fila["comando"]
        gt = fila_a_ground_truth(fila)

        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT_RECONSTRUIDO},
            {"role": "user", "content": construir_prompt_usuario(comando)},
        ]
        # return_dict=True es necesario: sin esto, algunas versiones de
        # transformers devuelven un BatchEncoding sin atributo .shape en
        # vez del tensor de input_ids, y model.generate() falla con
        # AttributeError al intentar leer inputs_tensor.shape[0].
        entrada = tokenizer.apply_chat_template(
            mensajes, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )

        inicio = time.perf_counter()
        with torch.no_grad():
            salida = modelo.generate(
                **entrada,          # unpackea input_ids + attention_mask
                max_new_tokens=128,
                do_sample=False,   # decodificación determinista (greedy),
                temperature=None,  # Sección 3.1: "sin muestreo, temperatura
                top_p=None,        # no aplicada"
                pad_token_id=tokenizer.eos_token_id,
            )
        latencia_s = time.perf_counter() - inicio

        texto_generado = tokenizer.decode(
            salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred, json_valido, nota = extraer_json(texto_generado)
        matches = comparar_campos(pred, gt)

        filas_resultado.append({
            "modelo": modelo_info.nombre,
            "idx": idx,
            "comando": comando,
            "gt": json.dumps(gt, ensure_ascii=False),
            "pred_raw": texto_generado,
            "pred_json": json.dumps(pred, ensure_ascii=False) if pred else "",
            "json_valido": json_valido,
            "parse_note": nota,
            "latencia_s": round(latencia_s, 3),
            **matches,
        })
        print(f"  [{idx:2d}] exact={matches['match_exact']}  "
              f"latencia={latencia_s:.2f}s  comando={comando[:40]!r}")

    del modelo, tokenizer
    return filas_resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--salida",
        default=str(RAIZ / "data" / "resultados_reproducidos.csv"),
        help="Ruta del CSV de detalle a generar",
    )
    parser.add_argument(
        "--modelos",
        nargs="*",
        default=None,
        help="Subconjunto de nombres de modelo a correr (por defecto, los 4)",
    )
    args = parser.parse_args()

    dataset = cargar_dataset()
    modelos_a_correr = MODELOS
    if args.modelos:
        modelos_a_correr = [m for m in MODELOS if m.nombre in args.modelos]

    todas_las_filas = []
    for modelo_info in modelos_a_correr:
        todas_las_filas.extend(evaluar_modelo(modelo_info, dataset))

    df_resultado = pd.DataFrame(todas_las_filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.salida, index=False)
    print(f"\nListo. {len(df_resultado)} filas escritas en {args.salida}")
    print("Corré src/metrics.py sobre ese CSV para generar el resumen agregado.")


if __name__ == "__main__":
    main()
