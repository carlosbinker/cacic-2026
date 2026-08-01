#!/usr/bin/env python3
"""
Validación de fidelidad metodológica: confirma que la lógica de
agregación de métricas (metrics.py) reproduce EXACTAMENTE el resumen
publicado en el paper (Tabla 2), partiendo del CSV de detalle real de
128 corridas (data/resultados_experimento_detalle.csv).

Esto no re-ejecuta los modelos (ver docstring de run_evaluation.py sobre
por qué el script original no está disponible) -- valida que, dado el
mismo detalle real, el código de scoring de este repo llega al mismo
resumen agregado que ya está en data/resultados_experimento_resumen.json.
Es la evidencia de que la reconstrucción es metodológicamente fiel, no
solo una afirmación.

Uso:
    python src/validar_contra_resultados_originales.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import calcular_resumen  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DETALLE_REAL = RAIZ / "data" / "resultados_experimento_detalle.csv"
RESUMEN_REAL = RAIZ / "data" / "resultados_experimento_resumen.json"

# Campos que sí se pueden validar a partir del CSV de detalle (load_time_s
# no está en ese CSV -- es tiempo de carga del modelo, no de un comando
# individual -- así que se excluye de la comparación).
CAMPOS_A_VALIDAR = [
    "n", "json_valido_pct", "exact_match_pct", "avg_latencia_s",
    "acc_intent_pct", "acc_dispositivo_pct", "acc_ubicacion_pct",
    "acc_valor_pct", "acc_unidad_pct",
]


def main():
    df_detalle = pd.read_csv(DETALLE_REAL)
    resumen_generado = {fila["modelo"]: fila for fila in calcular_resumen(df_detalle)}

    with open(RESUMEN_REAL, encoding="utf-8") as f:
        resumen_real = {fila["modelo"]: fila for fila in json.load(f)}

    ok = True
    for nombre_modelo, fila_real in resumen_real.items():
        fila_generada = resumen_generado.get(nombre_modelo)
        if fila_generada is None:
            print(f"FALTA: {nombre_modelo} no está en el resumen generado")
            ok = False
            continue

        for campo in CAMPOS_A_VALIDAR:
            real = fila_real[campo]
            generado = fila_generada[campo]
            if real != generado:
                print(f"DIFERENCIA [{nombre_modelo}.{campo}]: "
                      f"real={real}  generado={generado}")
                ok = False

    if ok:
        print("OK: el resumen generado por metrics.py coincide exactamente "
              "con data/resultados_experimento_resumen.json en los "
              f"{len(CAMPOS_A_VALIDAR)} campos validables por modelo.")
        print("Esto confirma que la lógica de scoring reconstruida en este "
              "repo es fiel a la que produjo los resultados del paper.")
    else:
        print("\nHay diferencias -- revisar metrics.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
