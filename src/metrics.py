#!/usr/bin/env python3
"""
Agregación de métricas: toma un CSV de detalle (una fila por comando x
modelo, con las columnas match_* y json_valido) y produce el resumen por
modelo -- las cinco métricas de la Sección 3.4 del paper: tasa de JSON
válido, coincidencia exacta, exactitud por campo, y latencia promedio.

Este módulo es la pieza que se valida contra los resultados reales del
estudio en validar_contra_resultados_originales.py: sin necesidad de
volver a correr los cuatro modelos (que requiere descargar varios GB de
pesos), se puede confirmar que la lógica de scoring reproduce exactamente
el mismo resumen agregado a partir del detalle real ya existente en
data/resultados_experimento_detalle.csv.

Uso:
    python src/metrics.py --detalle data/resultados_experimento_detalle.csv \
        --salida data/resumen_generado.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# Parámetros de cada modelo (Tabla 1 del paper) -- no están en el CSV de
# detalle, así que se agregan acá para que el resumen tenga el mismo
# esquema que data/resultados_experimento_resumen.json.
PARAMS_B_POR_MODELO = {
    "SmolLM2-360M-Instruct": 0.36,
    "Qwen2.5-0.5B-Instruct": 0.494,
    "Qwen2.5-1.5B-Instruct": 1.54,
    "SmolLM2-1.7B-Instruct": 1.71,
}


def calcular_resumen(df_detalle: pd.DataFrame) -> list[dict]:
    resumen = []
    orden_modelos = list(PARAMS_B_POR_MODELO.keys())

    for nombre_modelo in orden_modelos:
        df_m = df_detalle[df_detalle["modelo"] == nombre_modelo]
        if df_m.empty:
            continue

        n = len(df_m)
        resumen.append({
            "modelo": nombre_modelo,
            "params_b": PARAMS_B_POR_MODELO[nombre_modelo],
            "n": n,
            "json_valido_pct": round(100 * df_m["json_valido"].mean(), 1),
            "exact_match_pct": round(100 * df_m["match_exact"].mean(), 1),
            "avg_latencia_s": round(df_m["latencia_s"].mean(), 3),
            "acc_intent_pct": round(100 * df_m["match_intent"].mean(), 1),
            "acc_dispositivo_pct": round(100 * df_m["match_dispositivo"].mean(), 1),
            "acc_ubicacion_pct": round(100 * df_m["match_ubicacion"].mean(), 1),
            "acc_valor_pct": round(100 * df_m["match_valor"].mean(), 1),
            "acc_unidad_pct": round(100 * df_m["match_unidad"].mean(), 1),
        })
    return resumen


def calcular_taxonomia_errores(df_detalle: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la Tabla 4 del paper (taxonomía de errores) a partir
    del detalle: cuenta, por modelo, cuántas respuestas incorrectas caen
    en cada categoría de error (no excluyentes)."""
    filas = []
    for nombre_modelo in PARAMS_B_POR_MODELO:
        df_m = df_detalle[df_detalle["modelo"] == nombre_modelo]
        incorrectas = df_m[~df_m["match_exact"]]
        filas.append({
            "modelo": nombre_modelo,
            "total_incorrectas": len(incorrectas),
            "confusion_intencion": int((~incorrectas["match_intent"]).sum()),
            "confusion_dispositivo": int((~incorrectas["match_dispositivo"]).sum()),
            "confusion_ubicacion": int((~incorrectas["match_ubicacion"]).sum()),
            "confusion_valor_o_unidad": int(
                (~incorrectas["match_valor"] | ~incorrectas["match_unidad"]).sum()
            ),
        })
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detalle", default=str(RAIZ / "data" / "resultados_experimento_detalle.csv"))
    parser.add_argument("--salida", default=str(RAIZ / "data" / "resumen_generado.json"))
    args = parser.parse_args()

    df_detalle = pd.read_csv(args.detalle)
    resumen = calcular_resumen(df_detalle)

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Resumen agregado escrito en {args.salida}")
    for fila in resumen:
        print(f"  {fila['modelo']:28s}  exact_match={fila['exact_match_pct']:5.1f}%  "
              f"json_valido={fila['json_valido_pct']:5.1f}%  latencia={fila['avg_latencia_s']:.2f}s")


if __name__ == "__main__":
    main()
