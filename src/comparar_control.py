#!/usr/bin/env python3
"""Comparación de TRES BANDAS para los anclajes de continuidad (Delta 04 / RF19).

Compara, por anclaje: la cifra PUBLICADA (Tabla 2 original; fuente única
`data/resultados_experimento_resumen.json`, F0 congelado -- nunca se
hardcodea un número que ya vive ahí), la corrida de CONTROL con el prompt
original (`data/2026/control_prompt_original/`, mismo harness 2026, prompt
idéntico al publicado) y la corrida con el prompt NUEVO 2026
(`data/2026/detalle/`).

La banda de control aísla el efecto del prompt: si control ≈ publicado, el
harness reproduce el paper y la diferencia nuevo-vs-publicado es atribuible
al endurecimiento del prompt (RF1), no a un defecto del harness. Si control
diverge de publicado, hay un defecto del harness que hay que encontrar ANTES
de escribir la Sección 4 del paper reescrito (ver TODO.md, Delta 04): la
corrida de control es BLOQUEANTE para esa sección.

Uso:
    python src/comparar_control.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from models_2026 import slug
from run_control_prompt_original import ANCLAS_CONTROL

RAIZ = Path(__file__).resolve().parent.parent
RESUMEN_PUBLICADO_PATH = RAIZ / "data" / "resultados_experimento_resumen.json"
DIR_DETALLE_NUEVO = RAIZ / "data" / "2026" / "detalle"
DIR_CONTROL = RAIZ / "data" / "2026" / "control_prompt_original"
SALIDA_PATH = DIR_CONTROL / "comparacion_tres_bandas.json"

NOTA_LATENCIA = (
    "La latencia promedio NO es comparable entre bandas cuando la máquina no "
    "estuvo ociosa: las cifras publicadas provienen exclusivamente del barrido "
    "principal (data/2026/detalle/), corrido sobre host ocioso (RNF1); la "
    "corrida de control puede haber compartido la máquina con otro trabajo del "
    "orquestador mientras corría. Se reporta igual, con el mismo criterio que "
    "RF19, pero no debe usarse para argumentar un efecto del prompt sobre la "
    "latencia -- solo la exactitud estricta y la tasa de JSON válido lo permiten."
)


def cargar_publicado(ruta: Path = RESUMEN_PUBLICADO_PATH) -> dict[str, dict[str, Any]]:
    """Cifras de la Tabla 2 publicada, indexadas por nombre de modelo. Fuente
    única F0: nunca se hardcodea un número que ya vive en este archivo."""
    filas = json.loads(ruta.read_text(encoding="utf-8"))
    return {fila["modelo"]: fila for fila in filas}


def metricas_de_detalle(df: pd.DataFrame) -> dict[str, float]:
    """`exact_match_pct` / `json_valido_pct` / `avg_latencia_s` a partir de un
    CSV de detalle (control o nuevo), con el mismo criterio que
    `src/metrics.py` aplica al legacy: proporción de `True` * 100, promedio
    simple de latencia."""
    return {
        "n": int(len(df)),
        "exact_match_pct": round(100 * df["match_exact"].mean(), 1),
        "json_valido_pct": round(100 * df["json_valido"].mean(), 1),
        "avg_latencia_s": round(df["latencia_s"].mean(), 3),
    }


def cargar_banda(directorio: Path, nombre_modelo: str) -> dict[str, float] | None:
    """Métricas de un anclaje en una banda (control o nuevo), o `None` si el
    CSV todavía no existe (esa banda está incompleta para ese modelo)."""
    ruta = directorio / f"{slug(nombre_modelo)}.csv"
    if not ruta.exists():
        return None
    return metricas_de_detalle(pd.read_csv(ruta))


def _delta_pp(a: float | None, b: float | None) -> float | None:
    """Diferencia en puntos porcentuales de `a` respecto de `b` (a - b), o
    `None` si falta cualquiera de los dos lados."""
    if a is None or b is None:
        return None
    return round(a - b, 1)


def comparar_anclaje(nombre_modelo: str, publicado: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Arma la comparación de tres bandas de un anclaje."""
    if nombre_modelo not in publicado:
        raise ValueError(
            f"{nombre_modelo} no tiene cifra publicada en "
            f"{RESUMEN_PUBLICADO_PATH.name}; no es un anclaje de continuidad válido."
        )
    fila_publicada = publicado[nombre_modelo]
    pub = {
        "exact_match_pct": fila_publicada["exact_match_pct"],
        "json_valido_pct": fila_publicada["json_valido_pct"],
        "avg_latencia_s": fila_publicada["avg_latencia_s"],
    }
    control = cargar_banda(DIR_CONTROL, nombre_modelo)
    nuevo = cargar_banda(DIR_DETALLE_NUEVO, nombre_modelo)

    def _campo(banda: dict[str, float] | None, campo: str) -> float | None:
        return banda[campo] if banda else None

    return {
        "modelo": nombre_modelo,
        "publicado": pub,
        "control_prompt_original": control,
        "nuevo_prompt_2026": nuevo,
        "deltas_exact_match_pp": {
            "control_vs_publicado": _delta_pp(
                _campo(control, "exact_match_pct"), pub["exact_match_pct"]),
            "nuevo_vs_publicado": _delta_pp(
                _campo(nuevo, "exact_match_pct"), pub["exact_match_pct"]),
            "nuevo_vs_control": _delta_pp(
                _campo(nuevo, "exact_match_pct"), _campo(control, "exact_match_pct")),
        },
        "deltas_json_valido_pp": {
            "control_vs_publicado": _delta_pp(
                _campo(control, "json_valido_pct"), pub["json_valido_pct"]),
            "nuevo_vs_publicado": _delta_pp(
                _campo(nuevo, "json_valido_pct"), pub["json_valido_pct"]),
            "nuevo_vs_control": _delta_pp(
                _campo(nuevo, "json_valido_pct"), _campo(control, "json_valido_pct")),
        },
    }


def comparar_todo(publicado: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """La comparación de los tres anclajes, en el orden de `ANCLAS_CONTROL`."""
    if publicado is None:
        publicado = cargar_publicado()
    return {
        "anclajes": [comparar_anclaje(nombre, publicado) for nombre in ANCLAS_CONTROL],
        "nota_latencia": NOTA_LATENCIA,
    }


def _formatear_valor(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.1f}"


def imprimir_tabla(comparacion: dict[str, Any]) -> None:
    """Tabla markdown legible en stdout: exactitud estricta, las tres bandas
    y sus tres deltas, por anclaje."""
    print("| modelo | publicado | control (prompt original) | nuevo (prompt 2026) "
          "| Δ control-publicado | Δ nuevo-publicado | Δ nuevo-control |")
    print("|---|---|---|---|---|---|---|")
    for anclaje in comparacion["anclajes"]:
        control = anclaje["control_prompt_original"]
        nuevo = anclaje["nuevo_prompt_2026"]
        d = anclaje["deltas_exact_match_pp"]
        print(
            f"| {anclaje['modelo']} "
            f"| {_formatear_valor(anclaje['publicado']['exact_match_pct'])} "
            f"| {_formatear_valor(control['exact_match_pct'] if control else None)} "
            f"| {_formatear_valor(nuevo['exact_match_pct'] if nuevo else None)} "
            f"| {_formatear_valor(d['control_vs_publicado'])} "
            f"| {_formatear_valor(d['nuevo_vs_publicado'])} "
            f"| {_formatear_valor(d['nuevo_vs_control'])} |"
        )
    print()
    print(f"NOTA sobre latencia: {comparacion['nota_latencia']}")


def escribir_comparacion(comparacion: dict[str, Any], salida: Path) -> None:
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(comparacion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    comparacion = comparar_todo()
    escribir_comparacion(comparacion, SALIDA_PATH)
    imprimir_tabla(comparacion)
    return 0


if __name__ == "__main__":
    sys.exit(main())
