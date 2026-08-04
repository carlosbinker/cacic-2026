#!/usr/bin/env python3
"""Comparación de baseline RE-MEDIDO para los 4 modelos del paper original
(Delta 05 / subtask 19).

Por qué existe -- leer antes de tocar este archivo:

Un control con el prompt ORIGINAL del paper (subtask 18) NO reprodujo la
Tabla 2 publicada para los tres anclajes de continuidad. La causa, con
evidencia: `requirements.txt` legacy fija `transformers>=4.46.0` SIN cota
superior y `data/resultados_experimento_detalle.csv` no tiene columna
`transformers_version` -- el entorno de la corrida original nunca quedó
fijado ni registrado, así que es irrecuperable. Es irreproducibilidad del
trabajo ORIGINAL, no un defecto del harness nuevo (`tests/test_metricas.py`
sigue verde: la ruta de scoring es fiel).

Consecuencia (la que implementa este módulo): las cifras publicadas dejan de
ser un baseline validado y pasan a referencia HISTÓRICA, no reproducida. El
paper compara en cambio contra un baseline RE-MEDIDO e internamente
consistente: los 4 modelos del paper original, medidos bajo el mismo
harness/prompt 2026, misma máquina, temperatura 0. Este módulo arma esa
comparación -- publicado (histórico) vs re-medido, con su delta -- y
determina por CÓDIGO cuál es el mejor original re-medido, para que la
comparación central del paper (mejor arquitectura 2026 vs mejor original
re-medido) no se afirme a mano.

Uso:
    python src/comparar_baseline.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from models_2026 import ModeloEvaluado2026, roster_baseline_original, slug

RAIZ = Path(__file__).resolve().parent.parent
RESUMEN_PUBLICADO_PATH = RAIZ / "data" / "resultados_experimento_resumen.json"
DIR_DETALLE = RAIZ / "data" / "2026" / "detalle"
DIR_BASELINE = RAIZ / "data" / "2026" / "baseline_original"
SALIDA_PATH = DIR_BASELINE / "comparacion_baseline.json"

# El modelo cuyo re-medido ancla la comparación central del paper (mejor
# arquitectura 2026 vs mejor original re-medido). No es el mejor de los 12 por
# definición externa: es el que la Sección de resultados ya usa como cierre
# del barrido principal, y su CSV vive en data/2026/detalle/ (comprometido,
# solo lectura), así que leerlo de ahí nunca lo hardcodea como número.
MODELO_ARQUITECTURA_2026 = "granite-4.0-1b"

NOTA_PUBLICADO = (
    "Referencia histórica, NO reproducida: un control con el prompt original del "
    "paper no reprodujo estas cifras (ver Delta 05, subtask 18/19). La causa es "
    "irreproducibilidad del entorno original (transformers sin cota superior, sin "
    "columna de versión registrada), no un defecto del harness 2026."
)


def cargar_publicado(ruta: Path = RESUMEN_PUBLICADO_PATH) -> dict[str, dict[str, Any]]:
    """Cifras de la Tabla 2 publicada, indexadas por modelo. Fuente única F0:
    nunca se hardcodea un número que ya vive en este archivo."""
    filas = json.loads(ruta.read_text(encoding="utf-8"))
    return {fila["modelo"]: fila for fila in filas}


def metricas_de_detalle(df: pd.DataFrame) -> dict[str, float]:
    """`exact_match_pct` / `json_valido_pct` / `avg_latencia_s` de un CSV de
    detalle, mismo criterio que `src/comparar_control.py` y `src/metrics.py`."""
    return {
        "n": int(len(df)),
        "exact_match_pct": round(100 * df["match_exact"].mean(), 1),
        "json_valido_pct": round(100 * df["json_valido"].mean(), 1),
        "avg_latencia_s": round(df["latencia_s"].mean(), 3),
    }


def ruta_remedida(modelo: ModeloEvaluado2026) -> Path:
    """Dónde vive el CSV re-medido de un modelo del baseline original: en
    `data/2026/detalle/` si ya está en el roster activo (los 3 anclajes,
    corridos por el barrido principal), o en `data/2026/baseline_original/`
    si es baseline-completion (hoy, `Qwen2.5-0.5B-Instruct`)."""
    directorio = DIR_DETALLE if modelo.activo else DIR_BASELINE
    return directorio / f"{slug(modelo.nombre)}.csv"


def cargar_remedido(modelo: ModeloEvaluado2026) -> dict[str, float] | None:
    """Métricas re-medidas de un modelo, o `None` si su CSV todavía no existe."""
    ruta = ruta_remedida(modelo)
    if not ruta.exists():
        return None
    return metricas_de_detalle(pd.read_csv(ruta))


def _delta_pp(remedido: float | None, publicado: float) -> float | None:
    """Diferencia en puntos porcentuales (re-medido - publicado), o `None` si
    el re-medido todavía no existe."""
    if remedido is None:
        return None
    return round(remedido - publicado, 1)


def comparar_modelo(nombre_modelo: str, publicado: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Publicado (histórico) vs re-medido para un modelo del baseline original."""
    baseline = {m.nombre: m for m in roster_baseline_original()}
    if nombre_modelo not in baseline:
        raise ValueError(
            f"{nombre_modelo} no es uno de los 4 modelos de roster_baseline_original(); "
            f"la comparación de baseline está restringida a ellos."
        )
    if nombre_modelo not in publicado:
        raise ValueError(
            f"{nombre_modelo} no tiene cifra publicada en {RESUMEN_PUBLICADO_PATH.name}."
        )
    modelo = baseline[nombre_modelo]
    fila_publicada = publicado[nombre_modelo]
    pub = {
        "exact_match_pct": fila_publicada["exact_match_pct"],
        "json_valido_pct": fila_publicada["json_valido_pct"],
        "nota": NOTA_PUBLICADO,
    }
    remedido = cargar_remedido(modelo)
    return {
        "modelo": nombre_modelo,
        "publicado": pub,
        "remedido": remedido,
        "delta_exact_match_pp": _delta_pp(
            remedido["exact_match_pct"] if remedido else None, pub["exact_match_pct"]),
        "delta_json_valido_pp": _delta_pp(
            remedido["json_valido_pct"] if remedido else None, pub["json_valido_pct"]),
    }


def _mejor_remedido(comparaciones: list[dict[str, Any]]) -> dict[str, Any] | None:
    """El original re-medido con mayor `exact_match_pct`, calculado por código
    (nunca afirmado a mano): `None` si ninguno tiene todavía su CSV."""
    candidatos = [c for c in comparaciones if c["remedido"] is not None]
    if not candidatos:
        return None
    mejor = max(candidatos, key=lambda c: c["remedido"]["exact_match_pct"])
    return {"modelo": mejor["modelo"], "exact_match_pct": mejor["remedido"]["exact_match_pct"]}


def _comparacion_central(mejor_remedido: dict[str, Any] | None) -> dict[str, Any] | None:
    """El contraste central del paper: mejor arquitectura 2026 (leída del CSV
    real del barrido principal, nunca hardcodeada) vs mejor original re-medido.
    `None` si `mejor_remedido` todavía no está disponible."""
    if mejor_remedido is None:
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
        "mejor_arquitectura_2026": arquitectura,
        "mejor_original_remedido": mejor_remedido,
        "delta_pp": round(
            arquitectura["exact_match_pct"] - mejor_remedido["exact_match_pct"], 1
        ),
    }


def comparar_todo(publicado: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """La comparación de los 4 modelos del baseline original, en orden de registro."""
    if publicado is None:
        publicado = cargar_publicado()
    modelos = [comparar_modelo(m.nombre, publicado) for m in roster_baseline_original()]
    mejor = _mejor_remedido(modelos)
    return {
        "modelos": modelos,
        "mejor_remedido": mejor,
        "comparacion_central": _comparacion_central(mejor),
        "nota_publicado": NOTA_PUBLICADO,
    }


def _formatear_valor(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.1f}"


def imprimir_tabla(comparacion: dict[str, Any]) -> None:
    """Tabla markdown legible en stdout: publicado (histórico) vs re-medido,
    por modelo, más el ganador y la comparación central."""
    print("| modelo | publicado (histórico, NO reproducido) | re-medido (prompt 2026) "
          "| Δ re-medido − publicado |")
    print("|---|---|---|---|")
    for fila in comparacion["modelos"]:
        remedido = fila["remedido"]
        print(
            f"| {fila['modelo']} "
            f"| {_formatear_valor(fila['publicado']['exact_match_pct'])} "
            f"| {_formatear_valor(remedido['exact_match_pct'] if remedido else None)} "
            f"| {_formatear_valor(fila['delta_exact_match_pp'])} |"
        )
    print()
    if comparacion["mejor_remedido"]:
        print(f"Mejor original re-medido: {comparacion['mejor_remedido']['modelo']} "
              f"({comparacion['mejor_remedido']['exact_match_pct']:.1f}%)")
    central = comparacion.get("comparacion_central")
    if central:
        print(
            f"Comparación central: {central['mejor_arquitectura_2026']['modelo']} "
            f"({central['mejor_arquitectura_2026']['exact_match_pct']:.1f}%) vs "
            f"{central['mejor_original_remedido']['modelo']} "
            f"({central['mejor_original_remedido']['exact_match_pct']:.1f}%) -- "
            f"Δ {central['delta_pp']:.1f} pp"
        )
    print()
    print(f"NOTA sobre la columna publicada: {comparacion['nota_publicado']}")


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
