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
import functools
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
    "granite-4.0-350m": "regresión de caché en 5.14.1",
    "Qwen3.5-0.8B": "arquitectura no reconocida en 4.57.6",
}

ARCHIVO_PROMPT_SISTEMA = "prompt_sistema.tex"

# Modo de prompting por modelo (tabla §2.1 del índice): todo el roster activo
# usa chat_template nativo del tokenizer.
_MODO_PROMPTING = "Chat template"

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
DIR_SALIDA = RAIZ / "paper" / "02_reescrito" / "tablas"

PATH_PUBLICADO = RAIZ / "data" / "resultados_experimento_resumen.json"  # F0: solo lectura

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
           filas: list[list[str]], nota: str = "", ancho_completo: bool = False) -> str:
    """Arma un bloque table+tabular con booktabs, en el estilo LNCS.

    `ancho_completo=True` envuelve el `tabular` en `\\resizebox{\\textwidth}{!}{...}`
    para las tablas que se salen del ancho de página (Tablas 3, 4 y 5: muchas
    columnas o texto largo en `motivo_pin`)."""
    cuerpo = [
        r"\toprule",
        " & ".join(encabezado) + r" \\",
        r"\midrule",
    ]
    cuerpo += [" & ".join(f) + r" \\" for f in filas]
    cuerpo += [r"\bottomrule"]
    tabular = [f"\\begin{{tabular}}{{{spec}}}", *cuerpo, r"\end{tabular}"]
    if ancho_completo:
        tabular = [r"\resizebox{\textwidth}{!}{%", *tabular, "}"]
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
    """Tabla 1: roster activo de 12 modelos, con parámetros, tier, familia y
    prompting, en el orden canónico del paper (item 3 de la revisión de PDF:
    el mismo orden de la Fig. 1, agrupado por tier y tamaño creciente)."""
    filas = [
        [
            _escapar(m.nombre),
            f"{m.params_b:.2f}B",
            m.tier,
            _escapar(_familia(m.hf_repo_id)),
            _MODO_PROMPTING,
        ]
        for m in orden_canonico(roster_activo())
    ]
    return _tabla(
        caption="Roster de modelos evaluados",
        label="tab:modelos",
        spec="lrlll",
        encabezado=["Modelo", "Parámetros", "Tier", "Familia", "Prompting"],
        filas=filas,
    )


@functools.lru_cache(maxsize=1)
def _publicado() -> dict[str, dict]:
    """{modelo: fila publicada}, leída una sola vez de data/resultados_experimento_resumen.json."""
    filas = json.loads(PATH_PUBLICADO.read_text(encoding="utf-8"))
    return {f["modelo"]: f for f in filas}


def _celdas_continuidad_2025(nombre_modelo: str) -> tuple[str, str]:
    """Devuelve (Estricta 2025, Latencia 2025) ya formateadas, o ('--', '--') si el
    modelo no está en el estudio publicado de 2025."""
    fila = _publicado().get(nombre_modelo)
    if fila is None:
        return "--", "--"
    return f"{fila['exact_match_pct']:.1f}", f"{fila['avg_latencia_s']:.2f}"


def tabla2_resultados(resumen: list[dict]) -> str:
    """Tabla 2: resultados globales 2026 (JSON válido, estricta, laxa, latencia) más
    las dos columnas de continuidad 2025 (enmienda F8/RF19b), en el orden
    canónico del paper (item 3 de la revisión de PDF)."""
    filas = []
    resumen_ordenado = sorted(
        resumen, key=lambda f: clave_orden_canonico(f["tier"], f["params_b"], f["modelo"])
    )
    for fila in resumen_ordenado:
        estricta_2025, latencia_2025 = _celdas_continuidad_2025(fila["modelo"])
        filas.append([
            _escapar(fila["modelo"]),
            f"{fila['json_valido_pct']:.1f}",
            f"{fila['exact_match_pct']:.1f}",
            f"{fila['exact_match_laxo_pct']:.1f}",
            f"{fila['avg_latencia_s']:.2f}",
            estricta_2025,
            latencia_2025,
        ])
    return _tabla(
        caption="Resultados globales por modelo",
        label="tab:globales",
        spec="lrrrrrr",
        encabezado=[
            "Modelo", "JSON válido (\\%)", "Estricta (\\%)", "Laxa (\\%)",
            "Latencia (s)", "Estricta 2025 (\\%)", "Latencia 2025 (s)",
        ],
        filas=filas,
        ancho_completo=True,
    )


def tabla3_por_categoria(df: pd.DataFrame) -> str:
    """Tabla 3: exactitud por categoría lingüística, una columna por modelo, en
    el orden canónico del paper (item 3 de la revisión de PDF: la Tabla 3
    sigue a la Fig. 1, ya aprobada visualmente, y no al revés)."""
    orden = {m.nombre: i for i, m in enumerate(orden_canonico(roster_activo()))}
    columnas_modelo = sorted(
        (c for c in df.columns if c not in ("categoria", "n")),
        key=lambda c: orden.get(c, len(orden)),
    )
    encabezado = ["Categoría (n)"] + [_escapar(c) for c in columnas_modelo]
    filas = []
    for _, fila in df.iterrows():
        nombre_cat = CATEGORIAS_DISPLAY.get(fila["categoria"], fila["categoria"])
        celda_cat = f"{_escapar(nombre_cat)} ({fila['n']})"
        filas.append([celda_cat] + [f"{fila[c]:.1f}" for c in columnas_modelo])
    spec = "l" + "r" * len(columnas_modelo)
    return _tabla(
        caption="Exactitud estricta por categoría lingüística",
        label="tab:categorias",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        ancho_completo=True,
    )


def tabla4_taxonomia(df: pd.DataFrame) -> str:
    """Tabla 4: taxonomía de 7 etiquetas de error, una FILA por modelo (orden
    canónico, item 3) y una columna por etiqueta más el total de
    incorrectas -- transpuesta (item 1 de la revisión de PDF) respecto de la
    versión anterior, que ponía un modelo por columna: con doce modelos esa
    orientación desbordaba el ancho de columna LNCS. Los encabezados de
    etiqueta van rotados 90° para que las 12 filas no exijan comprimir tanto
    el ancho de cada columna."""
    orden = {m.nombre: i for i, m in enumerate(orden_canonico(roster_activo()))}
    modelos = sorted(df["modelo"], key=lambda m: orden.get(m, len(orden)))
    encabezado = ["Modelo", "Incorrectas"] + [
        r"\rotatebox{90}{" + _escapar(ETIQUETAS_ERROR_DISPLAY[e]) + "}"
        for e in ETIQUETAS_ERROR
    ]
    filas = []
    for m in modelos:
        fila_df = df.loc[df["modelo"] == m].iloc[0]
        fila = [_escapar(m), str(int(fila_df["total_incorrectas"]))]
        fila += [str(int(fila_df[e])) for e in ETIQUETAS_ERROR]
        filas.append(fila)
    spec = "l" + "r" * (1 + len(ETIQUETAS_ERROR))
    return _tabla(
        caption="Distribución de errores por categoría de la taxonomía",
        label="tab:taxonomia",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        ancho_completo=True,
    )


def tabla5_versiones() -> str:
    """Tabla 5: matriz de versiones de transformers (RF5), transpuesta (item 1
    de la revisión de PDF): un modelo por columna (encabezado rotado 90°,
    orden canónico del item 3) y una fila por atributo, en vez de un modelo
    por fila con una columna `Motivo` en texto libre que sólo entraba con
    `p{4cm}` + `resizebox`. El motivo de cada pin necesario se resume acá en
    el lenguaje del paper, sin texto crudo de excepción ni rutas de archivo
    (item 5 de la revisión de PDF: esas rutas son ruido y, en un envío ciego,
    una filtración de estructura de directorios local); el motivo completo,
    con su evidencia y la ruta del log, sigue documentado en
    `docker/README.md`, que no forma parte del envío."""
    modelos = orden_canonico(roster_activo())
    encabezado = ["Atributo"] + [
        r"\rotatebox{90}{" + _escapar(m.nombre) + "}" for m in modelos
    ]
    grupo_de = {BASELINE_TRANSFORMERS: "A", TRANSFORMERS_5X: "B"}
    fila_grupo = ["Grupo \\texttt{transformers}"] + [grupo_de[m.transformers_pin] for m in modelos]
    fila_version = ["Versión resuelta"] + [_VERSION_RESUELTA[m.transformers_pin] for m in modelos]
    fila_necesario = ["¿Necesario?"] + ["Sí" if m.motivo_pin else "Heredado" for m in modelos]
    fila_motivo = ["Motivo (si necesario)"] + [
        _escapar(_MOTIVO_EXPLICACION_PAPER[m.nombre]) if m.motivo_pin else "—"
        for m in modelos
    ]
    filas = [fila_grupo, fila_version, fila_necesario, fila_motivo]
    nota = (
        f"Grupo A (\\texttt{{{_escapar(BASELINE_TRANSFORMERS)}}}, resuelve "
        f"{_VERSION_RESUELTA[BASELINE_TRANSFORMERS]}) y grupo B "
        f"(\\texttt{{{_escapar(TRANSFORMERS_5X)}}}, resuelve "
        f"{_VERSION_RESUELTA[TRANSFORMERS_5X]}) son necesarios y mutuamente "
        "excluyentes sobre el roster: no existe una única versión mayor que "
        "sirva para las 12 columnas. El motivo completo de cada pin necesario, "
        "con su evidencia y referencia de log, está documentado en "
        "\\texttt{docker/README.md}."
    )
    spec = "l" + "c" * len(modelos)
    return _tabla(
        caption="Pines de \\texttt{transformers} por modelo",
        label="tab:versiones",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        nota=nota,
        ancho_completo=True,
    )


_ANCHO_VERBATIM = 84  # columnas de \ttfamily\small que entran a ancho LNCS sin overfull


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
        r"{\small\ttfamily",
        r"\begin{verbatim}",
        sistema,
        r"\end{verbatim}",
        r"}",
        "",
        r"\paragraph{Prompt de usuario (uno por comando, tras el de sistema).}",
        r"{\small\ttfamily",
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
