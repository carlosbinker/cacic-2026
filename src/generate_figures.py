#!/usr/bin/env python3
"""
Regenera las figuras 1 y 2 del paper a partir del resumen de resultados:
- Fig. 1: coincidencia exacta y latencia promedio, por modelo.
- Fig. 2: exactitud por campo del esquema JSON, por modelo.

Uso:
    python src/generate_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
RESUMEN_PATH = RAIZ / "data" / "resultados_experimento_resumen.json"
FIGURAS_DIR = RAIZ / "figures"

# Paleta simple y consistente entre ambas figuras.
COLOR_BARRA = "#4C72B0"
COLOR_LINEA = "#C44E52"


def nombres_cortos(nombre_modelo: str) -> str:
    return nombre_modelo.replace("-Instruct", "")


def generar_fig1(resumen: list[dict]):
    modelos = [nombres_cortos(m["modelo"]) for m in resumen]
    exact_match = [m["exact_match_pct"] for m in resumen]
    latencia = [m["avg_latencia_s"] for m in resumen]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(modelos, exact_match, color=COLOR_BARRA)
    ax1.set_title("Coincidencia exacta por modelo")
    ax1.set_ylabel("%")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=30)

    ax2.plot(modelos, latencia, marker="o", color=COLOR_LINEA)
    ax2.set_title("Latencia promedio de inferencia (CPU)")
    ax2.set_ylabel("segundos")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "fig1_exactitud_latencia.png", dpi=150)
    plt.close(fig)


def generar_fig2(resumen: list[dict]):
    modelos = [nombres_cortos(m["modelo"]) for m in resumen]
    campos = ["intent", "dispositivo", "ubicacion", "valor", "unidad"]
    claves = [f"acc_{c}_pct" for c in campos]

    fig, ax = plt.subplots(figsize=(9, 5))
    ancho = 0.15
    x = range(len(campos))

    for i, modelo in enumerate(modelos):
        valores = [resumen[i][clave] for clave in claves]
        posiciones = [xi + i * ancho for xi in x]
        ax.bar(posiciones, valores, width=ancho, label=modelo)

    ax.set_xticks([xi + 1.5 * ancho for xi in x])
    ax.set_xticklabels(campos)
    ax.set_ylabel("Exactitud (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exactitud por campo del esquema JSON, por modelo")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "fig2_exactitud_por_campo.png", dpi=150)
    plt.close(fig)


def main():
    with open(RESUMEN_PATH, encoding="utf-8") as f:
        resumen = json.load(f)

    FIGURAS_DIR.mkdir(exist_ok=True)
    generar_fig1(resumen)
    generar_fig2(resumen)
    print(f"Figuras regeneradas en {FIGURAS_DIR}/")


if __name__ == "__main__":
    main()
