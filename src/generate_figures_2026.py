#!/usr/bin/env python3
"""
Figuras del estudio 2026.

Módulo nuevo, no una edición de generate_figures.py: aquel está congelado
para que las figuras publicadas del paper original sigan siendo reproducibles.
Conserva su convención de CLI (--resumen / --sufijo).

Rediseño: barras horizontales agrupadas por tier. Con los 12 modelos del roster activo y nombres
como 'OLMo-2-0425-1B-Instruct', las barras verticales del diseño original
obligan a rotar las etiquetas hasta volverlas ilegibles; en horizontal el
nombre completo entra en el eje Y.

Dos salvaguardas de lectura, pensadas para impresión LNCS en escala de grises:

- Las barras de exactitud laxa cuyo modelo tiene respuestas laxamente
  correctas con JSON inválido (`laxo_json_invalido_n` > 0) se dibujan con
  textura rayada y se anotan con "*"; un pie de figura lista cuántas de esas
  respuestas son JSON inválido, para que un 93,8% laxo con formato roto no se
  lea como competencia real.
- Los pares de atención híbrida (nombre con "-h-" vs. su contraparte densa)
  se anotan en el panel de latencia con el cociente h/densa, calculado a
  partir del propio resumen -- nunca hardcodeado -- para que el hallazgo de
  la sobrecarga de la atención híbrida sea legible de un vistazo.

Uso:
    python src/generate_figures_2026.py [--resumen R] [--sufijo S] [--salida-dir D]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from models_2026 import ORDEN_TIERS, clave_orden_canonico  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PATH_RESUMEN = RAIZ / "data" / "2026" / "resumen_2026.json"
DIR_FIGURAS = RAIZ / "figures" / "2026"

SEPARACION_ENTRE_TIERS = 1.0   # en unidades de barra

CAMPOS_FIG2 = [
    ("acc_intent_pct", "intención"),
    ("acc_dispositivo_pct", "dispositivo"),
    ("acc_ubicacion_pct", "ubicación"),
    ("acc_valor_pct", "valor"),
    ("acc_unidad_pct", "unidad"),
]

# Orden categórico fijo para los 5 campos de fig. 2: un color por campo más
# una textura redundante de densidad mínima (un solo carácter), para que la
# serie se distinga sin depender del color -- pero sin producir moire a
# ancho de columna LNCS. Los 5 tonos son una rampa de luminancia monótona
# (evitando el par "casi idéntico" #63a0d4/#9fc6e8 del diseño anterior, que
# en escala de grises solo se distinguía por el rayado): luminancia
# aproximada 45 / 81 / 118 / 164 / 209 sobre 255, con saltos >35 entre
# vecinos.
ESTILOS_FIG2 = [
    {"color": "#123354", "hatch": None},
    {"color": "#205c97", "hatch": "/"},
    {"color": "#3585d4", "hatch": "x"},
    {"color": "#78ade2", "hatch": "."},
    {"color": "#bcd6f1", "hatch": "\\"},
]


def ordenar_para_grafico(resumen: list[dict]) -> list[dict]:
    """sub-1B primero, y dentro de cada tier por tamaño creciente.

    Delega en `models_2026.clave_orden_canonico` -- el mismo orden canónico
    que usa `generate_tex_tables.py` para la Tabla 3 y el resto de las tablas
    que listan modelos (RF/item 3 de la revisión de PDF), para que la figura y
    las tablas nunca puedan desincronizarse en el criterio de orden."""
    return sorted(
        resumen,
        key=lambda f: clave_orden_canonico(f["tier"], f["params_b"], f["modelo"]),
    )


def posiciones_con_separacion(tiers: list[str]) -> list[float]:
    """Posiciones en el eje Y, con un hueco visual al cambiar de tier."""
    posiciones, actual = [], 0.0
    for i, tier in enumerate(tiers):
        if i > 0 and tier != tiers[i - 1]:
            actual += SEPARACION_ENTRE_TIERS
        posiciones.append(actual)
        actual += 1.0
    return posiciones


def _validar(resumen: list[dict]) -> None:
    if not resumen:
        raise ValueError("El resumen está vacío: no hay nada que graficar")
    requeridas = {"modelo", "tier", "params_b", "exact_match_pct",
                  "exact_match_laxo_pct", "avg_latencia_s"}
    faltan = requeridas - set(resumen[0])
    if faltan:
        raise ValueError(f"Al resumen le faltan claves: {sorted(faltan)}")


def _etiquetas_de_tier(ax, tiers, posiciones):
    """Anota el nombre del tier a la altura de su primer modelo."""
    for tier in ORDEN_TIERS:
        indices = [i for i, t in enumerate(tiers) if t == tier]
        if indices:
            ax.text(-0.02, posiciones[indices[0]] - 0.7, tier,
                    transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=9, fontweight="bold", color="#444444")


def _marcar_laxo_con_json_invalido(ax, barras_laxa, resumen) -> list[str]:
    """Rayado + asterisco en las barras laxas infladas por JSON inválido.

    Devuelve las notas de pie de figura (una por modelo afectado), con los
    conteos leídos de `laxo_json_invalido_n` -- nunca hardcodeados.
    """
    notas = []
    for patch, f in zip(barras_laxa.patches, resumen):
        n_invalido = f.get("laxo_json_invalido_n", 0)
        if n_invalido > 0:
            patch.set_hatch("///")
            patch.set_edgecolor("#4a1010")
            patch.set_linewidth(0.7)
            ax.text(patch.get_width() + 1.0, patch.get_y() + patch.get_height() / 2,
                    "*", fontsize=10, fontweight="bold", va="center", color="#4a1010")
            n_total = f.get("n")
            detalle = f"{n_invalido}/{n_total}" if n_total else str(n_invalido)
            notas.append(f"{f['modelo']} ({detalle} JSON inválido)")
    return notas


def _anotar_penalizacion_hibrida(ax, pos, resumen) -> None:
    """Anota el cociente de latencia h/densa junto a cada variante híbrida.

    El par se identifica por convención de nombre ('-h-' vs. la misma cadena
    sin el segmento 'h-'); el cociente sale de `avg_latencia_s` del propio
    resumen, nunca hardcodeado.
    """
    latencia_por_modelo = {f["modelo"]: f["avg_latencia_s"] for f in resumen}
    for i, f in enumerate(resumen):
        if "-h-" not in f["modelo"]:
            continue
        contraparte = f["modelo"].replace("-h-", "-")
        if contraparte not in latencia_por_modelo:
            continue
        densa = latencia_por_modelo[contraparte]
        if not densa:
            continue
        cociente = f["avg_latencia_s"] / densa
        ax.text(f["avg_latencia_s"] + 1.0, pos[i], f"×{cociente:.2f} vs. densa",
                fontsize=7.5, va="center", color="#7a3300")


def generar_figuras(path_resumen: Path, dir_salida: Path, sufijo: str) -> list[Path]:
    resumen = ordenar_para_grafico(
        json.loads(Path(path_resumen).read_text(encoding="utf-8"))
    )
    _validar(resumen)

    nombres = [f["modelo"] for f in resumen]
    tiers = [f["tier"] for f in resumen]
    pos = posiciones_con_separacion(tiers)
    dir_salida.mkdir(parents=True, exist_ok=True)
    salidas = []

    # --- Fig. 1: exactitud (estricta vs laxa) | latencia ---
    alto = max(5.0, 0.42 * (max(pos) + 2))
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, alto))

    izq.barh([p - 0.2 for p in pos], [f["exact_match_pct"] for f in resumen],
              height=0.4, label="estricta", color="#2b6cb0")
    barras_laxa = izq.barh([p + 0.2 for p in pos],
                            [f["exact_match_laxo_pct"] for f in resumen],
                            height=0.4, label="laxa (semántica)", color="#90cdf4")
    notas_laxo = _marcar_laxo_con_json_invalido(izq, barras_laxa, resumen)
    izq.set_yticks(pos, nombres, fontsize=9)
    izq.invert_yaxis()
    izq.set_xlabel("Coincidencia exacta (%)")
    izq.set_xlim(0, 108)
    # Leyenda armada con parches proxy explícitos, no con
    # get_legend_handles_labels(): ese helper toma el estilo *actual* de los
    # artistas, y _marcar_laxo_con_json_invalido ya mutó dos de los parches
    # de "laxa (semántica)" (rayado + borde rojo) antes de este punto. Si el
    # primer parche mutado coincide con el primero del contenedor -- como
    # pasa con LFM2.5-230M -- la leyenda hereda ese estilo mutado para toda
    # la serie "laxa (semántica)", dejando dos entradas idénticas. Los
    # proxies fijan el estilo real de cada serie sin depender del orden ni
    # del estado mutable de los parches.
    handles = [
        mpatches.Patch(facecolor="#2b6cb0", label="estricta"),
        mpatches.Patch(facecolor="#90cdf4", label="laxa (semántica)"),
    ]
    if notas_laxo:
        handles.append(mpatches.Patch(
            facecolor="#90cdf4", hatch="///", edgecolor="#4a1010",
            linewidth=0.7, label="laxa con JSON inválido",
        ))
    # Leyenda debajo del área de datos (igual criterio que fig. 2): con 2
    # entradas "lower right" ya rozaba la barra laxa de granite-4.0-1b (100%,
    # el titular del paper); con 3 la caja creció y la tapaba.
    izq.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.08), fontsize=9, ncol=len(handles))
    izq.grid(axis="x", alpha=0.3)
    _etiquetas_de_tier(izq, tiers, pos)

    der.barh(pos, [f["avg_latencia_s"] for f in resumen], height=0.6, color="#dd6b20")
    der.set_yticks(pos, ["" for _ in nombres])
    der.invert_yaxis()
    der.set_xlabel("Latencia promedio en CPU (s)")
    der.grid(axis="x", alpha=0.3)
    _, xmax = der.get_xlim()
    der.set_xlim(0, xmax * 1.22)
    _anotar_penalizacion_hibrida(der, pos, resumen)

    if notas_laxo:
        fig.subplots_adjust(bottom=0.16)
        fig.text(
            0.01, 0.01,
            "* laxa con JSON inválido (el juez la acreditó por equivalencia "
            "semántica sobre una respuesta mal formada; no es un resultado "
            "limpio): " + "; ".join(notas_laxo) + ".",
            fontsize=7, wrap=True, ha="left", va="bottom",
        )
    else:
        fig.tight_layout()

    ruta1 = dir_salida / f"fig1_exactitud_latencia_2026{sufijo}.png"
    # bbox_inches="tight" recorta el lienzo al bbox real de los artistas ya
    # renderizados (incluida la etiqueta del tier a la izquierda del eje Y y
    # la nota de pie de figura cuando existe), en vez de depender de un
    # margen fijo: eso es lo que evita que la rama con nota al pie
    # (subplots_adjust solo ajusta 'bottom') clipee la "O" inicial de las
    # etiquetas largas como 'OLMo-2-0425-1B-Instruct' en el eje Y.
    fig.savefig(ruta1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    salidas.append(ruta1)

    # --- Fig. 2: exactitud por campo del esquema ---
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.55 * (max(pos) + 2))))
    n = len(CAMPOS_FIG2)
    alto_barra = 0.8 / n
    for k, (clave, etiqueta) in enumerate(CAMPOS_FIG2):
        desplazamiento = (k - (n - 1) / 2) * alto_barra
        estilo = ESTILOS_FIG2[k % len(ESTILOS_FIG2)]
        ax.barh([p + desplazamiento for p in pos],
                [f[clave] for f in resumen], height=alto_barra, label=etiqueta,
                color=estilo["color"], hatch=estilo["hatch"],
                edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(pos, nombres, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Exactitud por campo (%)")
    ax.set_xlim(0, 100)
    # Leyenda fuera del área de datos (franja horizontal debajo del eje): con
    # 5 series "lower right" caía encima de las barras de
    # SmolLM2-1.7B-Instruct y tapaba datos. bbox_inches="tight" en el
    # savefig de más abajo expande el lienzo para incluirla entera.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=9, ncol=len(CAMPOS_FIG2))
    ax.grid(axis="x", alpha=0.3)
    _etiquetas_de_tier(ax, tiers, pos)

    fig.tight_layout()
    ruta2 = dir_salida / f"fig2_exactitud_por_campo_2026{sufijo}.png"
    fig.savefig(ruta2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    salidas.append(ruta2)

    return salidas


def main() -> int:
    parser = argparse.ArgumentParser(description="Figuras del estudio 2026")
    parser.add_argument("--resumen", default=str(PATH_RESUMEN))
    parser.add_argument("--sufijo", default="")
    parser.add_argument("--salida-dir", default=str(DIR_FIGURAS))
    args = parser.parse_args()

    for ruta in generar_figuras(Path(args.resumen), Path(args.salida_dir), args.sufijo):
        print(f"escrita: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
