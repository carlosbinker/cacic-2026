#!/usr/bin/env python3
r"""
Figuras del estudio 2026.

Módulo nuevo, no una edición de generate_figures.py: aquel está congelado
para que las figuras publicadas del paper original sigan siendo reproducibles.
Conserva su convención de CLI (--resumen / --sufijo).

Rediseño de fig. 2: barras horizontales agrupadas por tier. Con los 12
modelos del roster activo y nombres como 'OLMo-2-0425-1B-Instruct', las
barras verticales del diseño original obligan a rotar las etiquetas hasta
volverlas ilegibles; en horizontal el nombre completo entra en el eje Y.

Rediseño de fig. 1 (2026-08-05, medido sobre un prototipo compilado): el
lienzo pasa a `figsize=(4.42, 2.4)` -- el ancho final impreso de LNCS a
`\includegraphics[width=0.92\textwidth]` -- para que el texto se autore ~1:1
y no quede por debajo del piso de legibilidad. La barra de exactitud (eje
inferior, 0-100%) se dibuja 100% apilada en vez de paralela: estricta, luego
el incremento de laxa con JSON válido, luego -- rayado -- el incremento de
laxa con JSON inválido (`laxo_json_invalido_n`, siempre subconjunto de la
laxa: `estricta ⊆ (laxa ∧ JSON válido) ⊆ laxa`), y el resto hasta 100% queda
sin relleno para que el blanco se lea como error. La barra de latencia
comparte el eje Y pero cuelga de un eje X propio arriba (`ax.twiny()`, 0-140
s). El orden ya no agrupa por tier: es exactitud estricta decreciente (ver
`ordenar_fig1_por_exactitud`), así que la separación de banda sub-1B/1-2B se
elimina sólo en esta figura -- la Tabla 1 conserva la suya. El pie de figura
con "*" y la anotación in-situ del cociente h/densa de latencia (atención
híbrida) del diseño anterior se retiran de esta figura: el primero es
redundante con la prosa de la Sección 4 (`laxo_json_invalido_n` por modelo ya
está citado ahí) y el segundo con la Sección 5
(`sec:discusion-latencia-hibrida`, que cita los mismos cocientes en prosa).

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


def ordenar_fig1_por_exactitud(resumen: list[dict]) -> list[dict]:
    """Orden de fig. 1: exactitud estricta decreciente; empate por laxa
    decreciente y luego alfabético.

    Clave de orden explícita -- nunca el orden de iteración de un diccionario
    ni de `roster_activo()` -- para que la figura sea reproducible byte a
    byte sobre cualquier `resumen_2026.json` con los mismos valores."""
    return sorted(
        resumen,
        key=lambda f: (-f["exact_match_pct"], -f["exact_match_laxo_pct"], f["modelo"]),
    )


def _segmentos_apilados_exactitud(resumen: list[dict]) -> dict[str, list[float]]:
    """Los tres incrementos de la barra 100% apilada de exactitud de fig. 1.

    `estricta ⊆ (laxa ∧ JSON válido) ⊆ laxa` para los 12 modelos del roster
    activo (n=32 constante): los segmentos son INCREMENTOS sobre el segmento
    anterior, nunca valores absolutos superpuestos. Un incremento negativo
    señalaría que esa inclusión no se sostiene en los datos -- un error de
    interpretación, no un detalle de dibujo -- así que aborta en vez de
    dibujar una barra aritméticamente incorrecta.
    """
    estricta, laxa_valida_incr, laxa_invalida_incr = [], [], []
    for f in resumen:
        a = f["exact_match_pct"]
        laxa = f["exact_match_laxo_pct"]
        n_total = f.get("n")
        n_invalido = f.get("laxo_json_invalido_n", 0)
        pct_invalido = (n_invalido / n_total * 100) if n_total else 0.0
        laxa_json_valida_abs = laxa - pct_invalido
        seg_valida = laxa_json_valida_abs - a
        seg_invalida = pct_invalido
        if seg_valida < -1e-6 or seg_invalida < -1e-6:
            raise ValueError(
                f"{f['modelo']}: segmento apilado negativo de fig. 1 "
                f"(estricta={a}, laxa_json_valida={laxa_json_valida_abs}, "
                f"laxa={laxa}, json_invalido_pct={pct_invalido}) -- no se "
                "cumple estricta ⊆ (laxa ∧ JSON válido) ⊆ laxa: revisar los "
                "datos de origen, esto no es un detalle de dibujo."
            )
        estricta.append(a)
        laxa_valida_incr.append(max(seg_valida, 0.0))
        laxa_invalida_incr.append(max(seg_invalida, 0.0))
    return {
        "estricta": estricta,
        "laxa_valida_incr": laxa_valida_incr,
        "laxa_invalida_incr": laxa_invalida_incr,
    }


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


def generar_figuras(path_resumen: Path, dir_salida: Path, sufijo: str) -> list[Path]:
    resumen_bruto = json.loads(Path(path_resumen).read_text(encoding="utf-8"))
    _validar(resumen_bruto)
    resumen = ordenar_para_grafico(resumen_bruto)

    nombres = [f["modelo"] for f in resumen]
    tiers = [f["tier"] for f in resumen]
    pos = posiciones_con_separacion(tiers)
    dir_salida.mkdir(parents=True, exist_ok=True)
    salidas = []

    # --- Fig. 1: exactitud 100% apilada (eje inferior) | latencia (eje
    # superior), mismo eje Y, orden propio (exactitud estricta decreciente,
    # sin separación de banda) -- ver docstring del módulo para el rediseño.
    resumen1 = ordenar_fig1_por_exactitud(resumen_bruto)
    nombres1 = [f["modelo"] for f in resumen1]
    pos1 = list(range(len(resumen1)))
    seg = _segmentos_apilados_exactitud(resumen1)

    FUENTE_MODELO, FUENTE_EJE, FUENTE_LEYENDA = 7.5, 8.0, 7.0
    ALTO_BARRA, DESPLAZ = 0.38, 0.23
    # Relleno de eje Y explícito (en vez de dejar el autoscale + invert_yaxis
    # por defecto): un colchón modesto y simétrico alcanza porque la leyenda
    # ya no vive dentro del área de datos (ver más abajo, corrección
    # 2026-08-05: la leyenda in-situ se superponía con las barras y consigo
    # misma -- se mueve fuera de los ejes, debajo del label "Exactitud").
    PAD_SUPERIOR, PAD_INFERIOR = 0.5, 0.5
    # Alto de lienzo mayor que el original (2.4in) para pagar, sin recortar
    # fuente ni leyenda, el espacio de la leyenda externa: ancho fijo al
    # impreso, alto el que la leyenda exige.
    fig1, ax_acc = plt.subplots(figsize=(4.42, 2.85), dpi=300)
    ax_lat = ax_acc.twiny()

    y_acc = [p + DESPLAZ for p in pos1]
    y_lat = [p - DESPLAZ for p in pos1]

    inicio_laxa_invalida = [a + b for a, b in zip(seg["estricta"], seg["laxa_valida_incr"])]
    ax_acc.barh(y_acc, seg["estricta"], height=ALTO_BARRA, left=0,
                color="#2b6cb0", zorder=2)
    ax_acc.barh(y_acc, seg["laxa_valida_incr"], height=ALTO_BARRA,
                left=seg["estricta"], color="#90cdf4", zorder=2)
    ax_acc.barh(y_acc, seg["laxa_invalida_incr"], height=ALTO_BARRA,
                left=inicio_laxa_invalida, color="#90cdf4", hatch="////",
                edgecolor="#4a1010", linewidth=0.5, zorder=3)
    # Contenedor sin relleno que delimita la barra completa (0-100%),
    # dibujado último para que su borde quede nítido sobre los rellenos: el
    # tramo que queda en blanco hasta el borde derecho ES la proporción de
    # error, no un margen decorativo.
    ax_acc.barh(y_acc, [100.0] * len(pos1), height=ALTO_BARRA, left=0,
                facecolor="none", edgecolor="#666666", linewidth=0.6, zorder=4)

    ax_lat.barh(y_lat, [f["avg_latencia_s"] for f in resumen1],
                height=ALTO_BARRA, color="#dd6b20", zorder=2)

    ax_acc.set_yticks(pos1, nombres1, fontsize=FUENTE_MODELO)
    ax_acc.tick_params(axis="y", length=0)
    # set_ylim con el límite mayor primero invierte el eje directamente (en
    # vez de invert_yaxis() sobre el autoscale): fija el colchón de abajo.
    ax_acc.set_ylim(len(pos1) - 1 + PAD_INFERIOR, -PAD_SUPERIOR)
    ax_acc.set_xlim(0, 100)
    ax_acc.set_xlabel("Exactitud (%)", fontsize=FUENTE_EJE)
    ax_acc.tick_params(axis="x", labelsize=FUENTE_EJE)

    ax_lat.set_xlim(0, 140)
    ax_lat.set_xlabel("Latencia (s)", fontsize=FUENTE_EJE)
    ax_lat.tick_params(axis="x", labelsize=FUENTE_EJE)

    # Leyenda con parches proxy explícitos (mismo motivo que en el diseño
    # anterior: los parches reales de la serie "laxa, JSON inv." tienen
    # ancho 0 en 10 de los 12 modelos, así que tomar el estilo de un patch
    # real al azar sería frágil). Corrección 2026-08-05: la leyenda dentro
    # del área de datos se superponía con las barras y sus propias entradas
    # se pisaban entre sí a 5,5pt -- se mueve FUERA de los ejes, debajo del
    # label "Exactitud" (orden vertical: barras -> ticks -> label ->
    # leyenda), con `fig.legend` en vez de `ax.legend` para que ancle en
    # coordenadas de figura y no de ejes. Sin la entrada "error": el usuario
    # decidió que el blanco hasta 100% se lee solo, con la cláusula del
    # caption como única explicación textual.
    handles1 = [
        mpatches.Patch(facecolor="#2b6cb0", label="estricta"),
        mpatches.Patch(facecolor="#90cdf4", label="laxa, JSON OK"),
        mpatches.Patch(facecolor="#90cdf4", hatch="////", edgecolor="#4a1010",
                        linewidth=0.5, label="laxa, JSON inv."),
        mpatches.Patch(facecolor="#dd6b20", label="latencia"),
    ]
    fig1.legend(handles=handles1, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                bbox_transform=fig1.transFigure, fontsize=FUENTE_LEYENDA,
                ncol=len(handles1), handlelength=1.2, handletextpad=0.4,
                columnspacing=1.0, labelspacing=0.6, borderpad=0.5)

    # Márgenes ajustados para que entren, sin recorte: el nombre de modelo
    # más largo a la izquierda, los dos ejes X (ticks + label) arriba y
    # abajo, y ahora la leyenda externa por debajo del label "Exactitud" --
    # verificado con Figure.get_tightbbox() contra el lienzo de 4.42x2.85in.
    fig1.subplots_adjust(left=0.31, right=0.90, top=0.87, bottom=0.30)

    ruta1 = dir_salida / f"fig1_exactitud_latencia_2026{sufijo}.png"
    # Sin bbox_inches="tight": el lienzo se autora a propósito al ancho final
    # impreso (0.92*textwidth de LNCS), y recortar el bbox cambiaría el
    # tamaño en píxeles relativo al figsize, rompiendo la correspondencia 1:1
    # entre el punto tipográfico nominal y el efectivo ya impreso.
    fig1.savefig(ruta1, dpi=300)
    plt.close(fig1)
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
