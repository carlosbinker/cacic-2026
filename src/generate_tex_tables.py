#!/usr/bin/env python3
"""
Genera los fragmentos .tex de las tablas del paper reescrito.

Ningún número del paper se escribe a mano: cada tabla se emite desde los
artefactos de data/2026/ o desde el registro de modelos. Cambiar los datos y
regenerar debe bastar para que el PDF quede correcto.

Cada fragmento es un bloque \\begin{table}...\\end{table} autónomo, sin
preámbulo, apto para \\input desde main.tex.

Uso:
    python src/generate_tex_tables.py [--salida-dir paper/02_reescrito/tablas]
"""

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd

from models_2026 import (
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    TRANSFORMERS_5X,
    clave_orden_canonico,
    orden_canonico,
    roster_activo,
)
from prompt_2026 import SYSTEM_PROMPT_2026
from prompt import construir_prompt_usuario
from taxonomia_2026 import (
    CATEGORIAS_DISPLAY,
    ETIQUETAS_ERROR,
    ETIQUETAS_ERROR_DISPLAY,
)

# Versiones efectivamente resueltas por pip para cada grupo (F3): no se re-sondean
# acá, se citan como constantes documentadas junto con el pin en el indice/README.
_VERSION_RESUELTA = {
    BASELINE_TRANSFORMERS: "4.57.6",
    TRANSFORMERS_5X: "5.14.1",
}

# Familia legible a partir del prefijo del hf_repo_id (F3 §2.1 del índice).
_PREFIJO_A_FAMILIA = {
    "LiquidAI": "LiquidAI",
    "ibm-granite": "IBM Granite",
    "Qwen": "Qwen",
    "HuggingFaceTB": "HuggingFaceTB",
    "allenai": "AllenAI",
}

# Explicación breve, en el lenguaje del paper, del motivo de cada pin
# NECESARIO -- item 5 de la revisión de PDF: la Tabla de versiones citaba
# antes el texto crudo de la excepción (incluidas rutas de
# `.claude-scratch/logs/...`), que es ruido y, en un envío ciego, una
# filtración de estructura de directorios local. El texto completo con la
# excepción y la ruta del log sigue documentado en `docker/README.md` (no es
# parte del envío); acá sólo se explica la causa, sin ruta ni traza. Clave =
# nombre del modelo, para no depender de la identidad de los `_MOTIVO_PIN_*`
# privados de `models_2026`.
_MOTIVO_EXPLICACION_PAPER: dict[str, str] = {
    "LFM2.5-230M": "tokenizer no soportado en 4.57.6",
    "LFM2.5-350M": "íd. (mismo tokenizer)",
    "granite-4.0-350m": "regresión de caché en la versión 5.14.1",
    "Qwen3.5-0.8B": "arquitectura no reconocida en 4.57.6",
}

ARCHIVO_PROMPT_SISTEMA = "prompt_sistema.tex"

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
DIR_SALIDA = RAIZ / "paper" / "02_reescrito" / "tablas"

ARCHIVOS_TABLAS = [
    "tabla1_modelos.tex",
    "tabla2_resultados_globales.tex",
    "tabla3_por_categoria.tex",
    "tabla4_taxonomia.tex",
    "tabla5_versiones.tex",
]

_REEMPLAZOS = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]


def _escapar(texto: object) -> str:
    """Escapa los caracteres especiales de LaTeX en texto que viene de datos."""
    salida = str(texto)
    for viejo, nuevo in _REEMPLAZOS:
        salida = salida.replace(viejo, nuevo)
    return salida


def _tabla(caption: str, label: str, spec: str, encabezado: list[str],
           filas: list[list[str]], nota: str = "", ancho_completo: bool = False,
           fuente_pequena: bool = False, cortes: frozenset[int] = frozenset()) -> str:
    """Arma un bloque table+tabular con booktabs, en el estilo LNCS.

    `ancho_completo=True` envuelve el `tabular` en `\\resizebox{\\textwidth}{!}{...}`;
    reservado como último recurso (ninguna de las 5 tablas del paper lo usa
    hoy -- `resizebox` no tiene piso de tamaño de letra y fue la causa real de
    que las tablas transpuestas se vieran comprimidas/ilegibles). Preferí
    `fuente_pequena=True` (envuelve en `{\\small ... }`, sin escalar) cuando
    la tabla entra a ancho de columna pero apretada con el cuerpo de texto
    normal.

    `cortes` son índices de fila (0-based) DESPUÉS de los cuales va un
    `\\midrule`: agrupan filas sin gastar una columna entera en decir a qué
    grupo pertenece cada una."""
    cuerpo = [
        r"\toprule",
        " & ".join(encabezado) + r" \\",
        r"\midrule",
    ]
    for i, f in enumerate(filas):
        cuerpo.append(" & ".join(f) + r" \\")
        if i in cortes:
            cuerpo.append(r"\midrule")
    cuerpo += [r"\bottomrule"]
    tabular = [f"\\begin{{tabular}}{{{spec}}}", *cuerpo, r"\end{tabular}"]
    if ancho_completo:
        tabular = [r"\resizebox{\textwidth}{!}{%", *tabular, "}"]
    elif fuente_pequena:
        # \footnotesize y no \small: con las 5 tablas y el prompt íntegro en el
        # cuerpo, un escalón de cuerpo de letra en las tablas es ~0,25 páginas
        # frente al límite de 10 del CFP, y ninguna tabla pierde una fila.
        tabular = [r"{\footnotesize", *tabular, "}"]
    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        *tabular,
    ]
    if nota:
        lineas.append(f"\\\\[2pt]\\footnotesize{{{nota}}}")
    lineas.append(r"\end{table}")
    return "\n".join(lineas) + "\n"


def _familia(hf_repo_id: str) -> str:
    """Familia del modelo a partir del prefijo (organización) del hf_repo_id."""
    prefijo = hf_repo_id.split("/", 1)[0]
    return _PREFIJO_A_FAMILIA.get(prefijo, prefijo)


def tabla1_modelos() -> str:
    """Tabla 1: roster activo de 12 modelos, con parámetros y familia, en el
    orden canónico del paper (el mismo de la Fig. 1, agrupado por tier y
    tamaño creciente).

    Sin columna `Prompting`: era constante en las 12 filas y no distinguía a
    ningún modelo. Sin columna `Tier`: los dos tiers son bloques contiguos del
    orden canónico, así que un `\\midrule` entre el último sub-1B y el primero
    de 1--2B dice lo mismo sin gastar una columna."""
    modelos = orden_canonico(roster_activo())
    filas = [
        [_escapar(m.nombre), f"{m.params_b:.2f}B", _escapar(_familia(m.hf_repo_id))]
        for m in modelos
    ]
    tiers = [m.tier for m in modelos]
    cortes = frozenset(
        i for i in range(len(tiers) - 1) if tiers[i] != tiers[i + 1]
    )
    return _tabla(
        caption="Conjunto de modelos evaluados. La línea horizontal separa la "
                "franja sub-1B (arriba) de la franja 1--2B (abajo)",
        label="tab:modelos",
        spec="lrl",
        encabezado=["Modelo", "Parámetros", "Familia"],
        filas=filas,
        fuente_pequena=True,
        cortes=cortes,
    )


def tabla2_resultados(resumen: list[dict]) -> str:
    """Tabla 2: resultados globales 2026 (JSON válido, estricta, laxa, latencia),
    en el orden canónico del paper (item 3 de la revisión de PDF). Sin
    `resizebox` (criterio del item 1 de la revisión de PDF, extendido a las 5
    tablas): los encabezados se parten en \\shortstack de 2 líneas cortas para
    no desbordar la caja LNCS."""
    filas = []
    resumen_ordenado = sorted(
        resumen, key=lambda f: clave_orden_canonico(f["tier"], f["params_b"], f["modelo"])
    )
    for fila in resumen_ordenado:
        filas.append([
            _escapar(fila["modelo"]),
            f"{fila['json_valido_pct']:.1f}",
            f"{fila['exact_match_pct']:.1f}",
            f"{fila['exact_match_laxo_pct']:.1f}",
            f"{fila['avg_latencia_s']:.2f}",
        ])
    return _tabla(
        caption="Resultados globales por modelo",
        label="tab:globales",
        spec="lrrrr",
        encabezado=[
            "Modelo",
            r"\shortstack{JSON\\válido\\(\%)}",
            r"\shortstack{Estricta\\(\%)}",
            r"\shortstack{Laxa\\(\%)}",
            r"\shortstack{Latencia\\(s)}",
        ],
        filas=filas,
        fuente_pequena=True,
    )


def tabla3_por_categoria(df: pd.DataFrame) -> str:
    """Tabla 3: exactitud estricta por categoría lingüística, una FILA por
    modelo (orden canónico, item 3 de la revisión de PDF) y una columna por
    categoría, con su `n` en el encabezado -- transpuesta (corrección del
    item 1: la orientación con doce modelos en columnas exigía `resizebox`
    y comprimía los nombres de modelo hasta ilegibles; con sólo 3
    categorías como columnas entra sin `resizebox`)."""
    orden = {m.nombre: i for i, m in enumerate(orden_canonico(roster_activo()))}
    columnas_modelo_orig = [c for c in df.columns if c not in ("categoria", "n")]
    modelos = sorted(columnas_modelo_orig, key=lambda c: orden.get(c, len(orden)))
    # El encabezado de categoría ("Encendido / apagado simple (15)") es más
    # largo que cualquier dato de la columna, así que necesita envolver en
    # varias líneas -- pero la columna de datos debe quedar alineada a la
    # derecha por el punto decimal (6.7 vs 100.0). Con columnas `r` los datos
    # ya alinean bien; el encabezado se envuelve aparte, en su propia celda,
    # con `\multicolumn{1}{c}{\parbox[b]{2.2cm}{\centering ...}}` -- así la
    # columna en sí sigue siendo `r` para las filas de datos.
    encabezado = ["Modelo"] + [
        r"\multicolumn{1}{c}{\parbox[b]{2.2cm}{\centering "
        f"{_escapar(CATEGORIAS_DISPLAY.get(fila['categoria'], fila['categoria']))} ({fila['n']})"
        "}}"
        for _, fila in df.iterrows()
    ]
    filas = []
    for m in modelos:
        fila = [_escapar(m)] + [f"{fila_df[m]:.1f}" for _, fila_df in df.iterrows()]
        filas.append(fila)
    spec = "l" + "r" * len(df)
    return _tabla(
        caption="Exactitud estricta por categoría lingüística",
        label="tab:categorias",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        fuente_pequena=True,
    )


def tabla4_taxonomia(df: pd.DataFrame) -> str:
    """Tabla 4: taxonomía de 7 etiquetas de error, una FILA por modelo (orden
    canónico, item 3) y una columna por etiqueta más el total de
    incorrectas -- transpuesta (item 1 de la revisión de PDF) respecto de la
    versión anterior, que ponía un modelo por columna: con doce modelos esa
    orientación desbordaba el ancho de columna LNCS. Los encabezados de
    etiqueta van rotados 90° para que las 12 filas no exijan comprimir tanto
    el ancho de cada columna. Sin `resizebox` (corrección del item 1 de la
    revisión de PDF: `resizebox` no tiene piso de tamaño de letra): las
    columnas de enteros chicos con encabezados rotados entran a ancho de
    columna LNCS con una letra un escalón más chica.

    Las etiquetas que quedaron en cero para los doce modelos NO llevan
    columna: el paper sólo muestra lo que efectivamente aparece en los
    resultados, y una etiqueta que el juez pudo usar y nunca usó se reporta en
    una frase de prosa, no en una columna entera de ceros. Una etiqueta con
    una sola ocurrencia sí lleva columna --- es una medición, no un vacío."""
    orden = {m.nombre: i for i, m in enumerate(orden_canonico(roster_activo()))}
    modelos = sorted(df["modelo"], key=lambda m: orden.get(m, len(orden)))
    etiquetas = [e for e in ETIQUETAS_ERROR if int(df[e].sum()) > 0]
    encabezado = ["Modelo", "Incorrectas"] + [
        r"\rotatebox{90}{" + _escapar(ETIQUETAS_ERROR_DISPLAY[e]) + "}"
        for e in etiquetas
    ]
    filas = []
    for m in modelos:
        fila_df = df.loc[df["modelo"] == m].iloc[0]
        fila = [_escapar(m), str(int(fila_df["total_incorrectas"]))]
        fila += [str(int(fila_df[e])) for e in etiquetas]
        filas.append(fila)
    spec = "l" + "r" * (1 + len(etiquetas))
    return _tabla(
        caption="Distribución de errores por categoría de la taxonomía",
        label="tab:taxonomia",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        fuente_pequena=True,
    )


def tabla5_versiones() -> str:
    """Tabla 5: matriz de versiones de transformers (RF5), un modelo por FILA
    (orden canónico, item 3 de la revisión de PDF), con `Motivo` en una
    columna `p{}` que envuelve el texto -- corrección: la orientación con
    doce modelos en columnas (encabezados rotados 90°) dejaba la fila de
    `Motivo` en celdas de 1/13 del ancho, ilegible. El motivo de cada pin
    necesario se resume en el lenguaje del paper, sin texto crudo de
    excepción ni rutas de archivo: esas rutas son ruido y, en un envío ciego,
    una filtración de estructura de directorios local. Sin columna
    `¿Necesario?`: su valor `Heredado` no respondía la pregunta del
    encabezado y era redundante con `Motivo`, donde un motivo no vacío ya
    significa que ese modelo forzó el pin. Sin `resizebox`: entra a ancho de
    columna LNCS sin comprimir letra."""
    modelos = orden_canonico(roster_activo())
    grupo_de = {BASELINE_TRANSFORMERS: "A", TRANSFORMERS_5X: "B"}
    filas = []
    for m in modelos:
        version_resuelta = _VERSION_RESUELTA[m.transformers_pin]
        motivo = _escapar(_MOTIVO_EXPLICACION_PAPER[m.nombre]) if m.motivo_pin else "—"
        filas.append([
            _escapar(m.nombre),
            grupo_de[m.transformers_pin],
            version_resuelta,
            motivo,
        ])
    nota = (
        f"Grupo A (\\texttt{{{_escapar(BASELINE_TRANSFORMERS)}}}, resuelve "
        f"{_VERSION_RESUELTA[BASELINE_TRANSFORMERS]}) y grupo B "
        f"(\\texttt{{{_escapar(TRANSFORMERS_5X)}}}, resuelve "
        f"{_VERSION_RESUELTA[TRANSFORMERS_5X]}) son necesarios y mutuamente "
        "excluyentes sobre el conjunto evaluado: no existe una única versión mayor que "
        "sirva para las 12 filas activas. Un motivo en blanco (—) indica que "
        "el modelo no impuso ningún requisito propio y quedó en el grupo por "
        "omisión."
    )
    return _tabla(
        caption="Pines de \\texttt{transformers} por modelo",
        label="tab:versiones",
        spec="lllp{3.6cm}",
        encabezado=["Modelo", "Grupo", "Versión resuelta", "Motivo"],
        filas=filas,
        nota=nota,
        fuente_pequena=True,
    )


# Columnas de \ttfamily\footnotesize que entran a ancho LNCS sin overfull. El
# bloque pasó de \small a \footnotesize por presupuesto de páginas (límite de
# 10 del CFP): el prompt se sigue emitiendo íntegro desde la constante real,
# sólo baja un escalón de cuerpo, y el ancho de envoltura sube en proporción
# inversa (64 * 9/8 = 72) para no desperdiciar ancho de caja.
_ANCHO_VERBATIM = 72


def _envolver_para_verbatim(texto: str, ancho: int = _ANCHO_VERBATIM) -> str:
    """Envuelve `texto` a `ancho` columnas preservando los saltos de línea que
    ya trae (párrafos separados por línea en blanco, o el salto duro antes del
    JSON de ejemplo): sólo parte las líneas que de otro modo desbordarían el
    ancho de columna dentro de `verbatim`, nunca colapsa un salto existente."""
    parrafos = texto.split("\n\n")
    salida_parrafos = []
    for parrafo in parrafos:
        lineas = parrafo.split("\n")
        envueltas = [
            "\n".join(textwrap.wrap(linea, width=ancho)) if linea.strip() else linea
            for linea in lineas
        ]
        salida_parrafos.append("\n".join(envueltas))
    return "\n\n".join(salida_parrafos)


def prompt_sistema_tex() -> str:
    """Fragmento con el prompt de sistema y el prompt de usuario EXACTOS que usa
    el barrido (item 6 de la revisión de PDF, corrección de §3.3): se generan
    importando `SYSTEM_PROMPT_2026` (`src/prompt_2026.py`) y
    `construir_prompt_usuario` (`src/prompt.py`) -- nunca transcritos a mano
    -- para que el .tex no pueda desincronizarse en silencio de lo que
    efectivamente se ejecutó. Se muestran íntegros (no un resumen ni un diff
    contra ninguna versión anterior): es, sin más, el prompt de este trabajo."""
    sistema = _envolver_para_verbatim(SYSTEM_PROMPT_2026)
    usuario = construir_prompt_usuario("<comando del usuario>")
    lineas = [
        r"\paragraph{Prompt de sistema (texto completo).}",
        r"{\footnotesize\ttfamily",
        r"\begin{verbatim}",
        sistema,
        r"\end{verbatim}",
        r"}",
        "",
        r"\paragraph{Prompt de usuario (uno por comando, tras el de sistema).}",
        r"{\footnotesize\ttfamily",
        r"\begin{verbatim}",
        usuario,
        r"\end{verbatim}",
        r"}",
    ]
    return "\n".join(lineas) + "\n"


def _cargar_resumen_2026() -> list[dict]:
    return json.loads((DIR_2026 / "resumen_2026.json").read_text(encoding="utf-8"))


def _cargar_por_categoria() -> pd.DataFrame:
    return pd.read_csv(DIR_2026 / "exactitud_por_categoria.csv")


def _cargar_taxonomia() -> pd.DataFrame:
    return pd.read_csv(DIR_2026 / "taxonomia_2026.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida-dir", type=Path, default=DIR_SALIDA)
    args = parser.parse_args()

    args.salida_dir.mkdir(parents=True, exist_ok=True)

    fragmentos = {
        "tabla1_modelos.tex": tabla1_modelos(),
        "tabla2_resultados_globales.tex": tabla2_resultados(_cargar_resumen_2026()),
        "tabla3_por_categoria.tex": tabla3_por_categoria(_cargar_por_categoria()),
        "tabla4_taxonomia.tex": tabla4_taxonomia(_cargar_taxonomia()),
        "tabla5_versiones.tex": tabla5_versiones(),
    }
    for nombre in ARCHIVOS_TABLAS:
        ruta = args.salida_dir / nombre
        ruta.write_text(fragmentos[nombre], encoding="utf-8")
        print(ruta)

    # Fuera de ARCHIVOS_TABLAS (item 6): no es una de las 5 tablas numeradas,
    # es el fragmento del prompt de §3.3, mismo directorio y convención de
    # \input.
    ruta_prompt = args.salida_dir / ARCHIVO_PROMPT_SISTEMA
    ruta_prompt.write_text(prompt_sistema_tex(), encoding="utf-8")
    print(ruta_prompt)


if __name__ == "__main__":
    main()
