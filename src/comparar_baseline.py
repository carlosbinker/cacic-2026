#!/usr/bin/env python3
"""Matriz de prompt del brazo de generación ANTERIOR (Delta 06).

Por qué existe -- leer antes de tocar este archivo:

`paper_cacic_LNCS_word.docx` es un BORRADOR del paper que se está
reescribiendo, no trabajo publicado (corrige hacia adelante el Delta 04/05,
que lo trataba como si lo fuera, sin editarlos). Sus cifras
(SmolLM2-360M-Instruct 18.8, Qwen2.5-0.5B-Instruct 43.8, Qwen2.5-1.5B-Instruct
50.0, SmolLM2-1.7B-Instruct 59.4) son valores de BORRADOR SUPERADOS, no una
base de comparación: no hay "publicado vs re-medido" que reportar, y este
módulo NO es una auditoría de reproducibilidad -- esa framing queda
RETIRADA por completo (ver TODO.md, Delta 06).

Los 4 modelos de `roster_baseline_original()` son el brazo de generación
ANTERIOR de ESTE MISMO experimento: se miden bajo el mismo harness, prompt,
máquina, versiones fijadas y decodificación greedy que los modelos más
nuevos (Granite 4 / Qwen3.5 / LFM2.5 / OLMo-2), primero bajo el prompt
ORIGINAL del borrador (`data/2026/control_prompt_original/`) y después bajo
el prompt 2026 endurecido (`data/2026/detalle/` para los tres ya activos,
`data/2026/baseline_original/` para la baseline-completion de
`Qwen2.5-0.5B-Instruct`). Eso hace que la comparación generacional sea
internamente válida SIN apoyarse en ningún número del borrador. El control
y la baseline-completion sobreviven solo como justificación METODOLÓGICA de
por qué se re-midieron en vez de reusar las cifras del borrador -- nunca
como sección de resultados.

Este módulo arma la matriz 4×2 (4 modelos × {prompt original, prompt 2026})
de exactitud estricta y tasa de JSON válido, el efecto de prompt por modelo
(2026 − original), y calcula por CÓDIGO el titular del paper: la mejor
generación anterior BAJO EL MISMO PROMPT 2026 vs `granite-4.0-1b` (mejor
arquitectura 2026), para no afirmarlo a mano ni mezclar prompts -- ese error
real ya se cometió una vez (comparar una celda de prompt 2026 contra una de
prompt original produjo un delta de ~40 pp que no correspondía a nada real).

Las latencias de `control_prompt_original/` y `baseline_original/` se
midieron con la máquina compartida con otro trabajo del orquestador (no
ociosa): no son válidas para argumentar sobre latencia y por eso ni siquiera
se cargan en la celda. Las latencias del paper vienen exclusivamente de
`data/2026/detalle/`, corrido en secuencial estricto sobre máquina ociosa.

Uso:
    python src/comparar_baseline.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from models_2026 import ModeloEvaluado2026, roster_baseline_original, slug

RAIZ = Path(__file__).resolve().parent.parent
DIR_DETALLE = RAIZ / "data" / "2026" / "detalle"
DIR_BASELINE = RAIZ / "data" / "2026" / "baseline_original"
DIR_CONTROL = RAIZ / "data" / "2026" / "control_prompt_original"
SALIDA_PATH = RAIZ / "data" / "2026" / "matriz_generacion_anterior.json"

Prompt = Literal["original", "2026"]

# El modelo cuyo CSV real ancla el lado "arquitectura 2026" del titular. No es
# el mejor de los 12 por definición externa: es el que la Sección de
# resultados ya usa como cierre del barrido principal, y su CSV vive en
# data/2026/detalle/ (comprometido, solo lectura), así que leerlo de ahí nunca
# lo hardcodea como número.
MODELO_ARQUITECTURA_2026 = "granite-4.0-1b"

NOTA_LATENCIA = (
    "Las latencias de control_prompt_original/ y baseline_original/ se midieron con "
    "la maquina NO ociosa (trabajo concurrente del orquestador mientras corrian): no "
    "son validas para argumentar sobre latencia y por eso no se incluyen en esta "
    "matriz. Las latencias del paper provienen exclusivamente de data/2026/detalle/, "
    "corrido en secuencial estricto sobre maquina ociosa (RNF1)."
)


def metricas_de_detalle(df: pd.DataFrame) -> dict[str, float]:
    """`exact_match_pct` / `json_valido_pct` de un CSV de detalle. Sin
    latencia: ver `NOTA_LATENCIA` sobre por qué esta matriz no la reporta."""
    return {
        "n": int(len(df)),
        "exact_match_pct": round(100 * df["match_exact"].mean(), 1),
        "json_valido_pct": round(100 * df["json_valido"].mean(), 1),
    }


def ruta_celda(modelo: ModeloEvaluado2026, prompt: Prompt) -> Path:
    """Dónde vive el CSV de una celda (modelo, prompt) de la matriz 4×2.

    Prompt original: siempre `control_prompt_original/` (los 4 modelos se
    miden ahí, sin excepción). Prompt 2026: `detalle/` si el modelo ya está
    en el roster activo (los 3 anclajes), o `baseline_original/` si es
    baseline-completion (hoy, `Qwen2.5-0.5B-Instruct`)."""
    if prompt == "original":
        return DIR_CONTROL / f"{slug(modelo.nombre)}.csv"
    directorio = DIR_DETALLE if modelo.activo else DIR_BASELINE
    return directorio / f"{slug(modelo.nombre)}.csv"


def cargar_celda(modelo: ModeloEvaluado2026, prompt: Prompt) -> dict[str, float] | None:
    """Métricas de una celda, o `None` si su CSV todavía no existe (celda
    PENDIENTE, no un error: hoy es exactamente la celda (Qwen2.5-0.5B-Instruct,
    original), que el orquestador completa después de este reporte)."""
    ruta = ruta_celda(modelo, prompt)
    if not ruta.exists():
        return None
    return metricas_de_detalle(pd.read_csv(ruta))


def _delta_pp(nuevo: float | None, viejo: float | None) -> float | None:
    """Efecto de prompt (2026 − original) en puntos porcentuales, o `None` si
    falta cualquiera de las dos celdas."""
    if nuevo is None or viejo is None:
        return None
    return round(nuevo - viejo, 1)


def fila_modelo(modelo: ModeloEvaluado2026) -> dict[str, Any]:
    """Una fila de la matriz 4×2: las dos celdas de un modelo y su efecto de
    prompt (2026 − original), calculado solo si ambas celdas existen."""
    original = cargar_celda(modelo, "original")
    nuevo = cargar_celda(modelo, "2026")
    return {
        "modelo": modelo.nombre,
        "prompt_original": original,
        "prompt_2026": nuevo,
        "efecto_prompt_exact_match_pp": _delta_pp(
            nuevo["exact_match_pct"] if nuevo else None,
            original["exact_match_pct"] if original else None,
        ),
        "efecto_prompt_json_valido_pp": _delta_pp(
            nuevo["json_valido_pct"] if nuevo else None,
            original["json_valido_pct"] if original else None,
        ),
    }


def matriz() -> list[dict[str, Any]]:
    """La matriz 4×2 completa, en el orden de registro de
    `roster_baseline_original()`."""
    return [fila_modelo(m) for m in roster_baseline_original()]


def mejor_bajo_prompt(filas: list[dict[str, Any]], clave_prompt: str) -> dict[str, Any] | None:
    """El modelo de generación anterior con mayor `exact_match_pct` bajo UN
    prompt fijo (`clave_prompt` es `"prompt_original"` o `"prompt_2026"`),
    calculado por código; `None` si ninguna fila tiene todavía esa celda."""
    candidatos = [(f["modelo"], f[clave_prompt]) for f in filas if f[clave_prompt] is not None]
    if not candidatos:
        return None
    nombre, celda = max(candidatos, key=lambda par: par[1]["exact_match_pct"])
    return {"modelo": nombre, "exact_match_pct": celda["exact_match_pct"]}


def comparar_celdas(fila_a: dict[str, Any], prompt_a: Prompt,
                     fila_b: dict[str, Any], prompt_b: Prompt) -> float:
    """Diferencia de `exact_match_pct` entre dos celdas de la matriz (celda_a
    − celda_b). Guardia de sanidad OBLIGATORIA: rechaza comparar celdas de
    distinto prompt -- ese error real ya se cometió una vez (una celda de
    prompt 2026 contra una de prompt original) y produjo un delta de ~40 pp
    que no correspondía a nada real. También rechaza si alguna celda todavía
    no tiene CSV (pendiente): no hay nada que comparar todavía."""
    if prompt_a != prompt_b:
        raise ValueError(
            f"no se puede comparar la celda de prompt {prompt_a!r} contra la de "
            f"prompt {prompt_b!r}: el titular del paper se calcula SIEMPRE con el "
            f"prompt fijo, nunca mezclando prompts distintos."
        )
    celda_a = fila_a[f"prompt_{prompt_a}"]
    celda_b = fila_b[f"prompt_{prompt_b}"]
    if celda_a is None or celda_b is None:
        raise ValueError(
            "no se puede comparar: al menos una de las dos celdas todavía no tiene "
            "CSV (pendiente)."
        )
    return round(celda_a["exact_match_pct"] - celda_b["exact_match_pct"], 1)


def titular(filas: list[dict[str, Any]]) -> dict[str, Any] | None:
    """El contraste central del paper, prompt SIEMPRE fijo: mejor arquitectura
    2026 (leída de su propio CSV real de `data/2026/detalle/`) vs mejor
    generación anterior bajo el MISMO prompt 2026. `None` si esa celda de
    `granite-4.0-1b` o la mejor generación anterior todavía no existen."""
    mejor_anterior = mejor_bajo_prompt(filas, "prompt_2026")
    if mejor_anterior is None:
        return None
    ruta_arquitectura = DIR_DETALLE / f"{slug(MODELO_ARQUITECTURA_2026)}.csv"
    if not ruta_arquitectura.exists():
        return None
    metricas_arquitectura = metricas_de_detalle(pd.read_csv(ruta_arquitectura))
    arquitectura = {
        "modelo": MODELO_ARQUITECTURA_2026,
        "exact_match_pct": metricas_arquitectura["exact_match_pct"],
    }
    return {
        "prompt": "2026",
        "mejor_arquitectura_2026": arquitectura,
        "mejor_generacion_anterior": mejor_anterior,
        "delta_pp": round(
            arquitectura["exact_match_pct"] - mejor_anterior["exact_match_pct"], 1
        ),
    }


def comparar_todo() -> dict[str, Any]:
    """La matriz 4×2 completa más el titular del paper y la nota de latencia."""
    filas = matriz()
    return {
        "matriz": filas,
        "titular": titular(filas),
        "nota_latencia": NOTA_LATENCIA,
    }


def _formatear_valor(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.1f}"


def imprimir_tabla(comparacion: dict[str, Any]) -> None:
    """Tabla markdown legible en stdout: las dos celdas de la matriz y el
    efecto de prompt, por modelo, más el titular calculado."""
    print("| modelo | prompt original | prompt 2026 | Δ prompt (2026 − original) |")
    print("|---|---|---|---|")
    for fila in comparacion["matriz"]:
        original = fila["prompt_original"]
        nuevo = fila["prompt_2026"]
        celda_original = _formatear_valor(original["exact_match_pct"] if original else None)
        celda_nuevo = _formatear_valor(nuevo["exact_match_pct"] if nuevo else None)
        if original is None or nuevo is None:
            celda_original = "pendiente" if original is None else celda_original
            celda_nuevo = "pendiente" if nuevo is None else celda_nuevo
        print(
            f"| {fila['modelo']} | {celda_original} | {celda_nuevo} "
            f"| {_formatear_valor(fila['efecto_prompt_exact_match_pp'])} |"
        )
    print()
    titular_ = comparacion.get("titular")
    if titular_:
        print(
            f"Titular (prompt {titular_['prompt']}): "
            f"{titular_['mejor_arquitectura_2026']['modelo']} "
            f"({titular_['mejor_arquitectura_2026']['exact_match_pct']:.1f}%) vs "
            f"{titular_['mejor_generacion_anterior']['modelo']} "
            f"({titular_['mejor_generacion_anterior']['exact_match_pct']:.1f}%) -- "
            f"Δ {titular_['delta_pp']:.1f} pp"
        )
    else:
        print("Titular: pendiente (falta al menos una celda bajo el prompt 2026).")
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
