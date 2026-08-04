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
from pathlib import Path

import pandas as pd

from models_2026 import (
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    TRANSFORMERS_5X,
    roster_activo,
)
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
    """Tabla 1: roster activo de 12 modelos, con parámetros, tier, familia y prompting."""
    filas = [
        [
            _escapar(m.nombre),
            f"{m.params_b:.2f}B",
            m.tier,
            _escapar(_familia(m.hf_repo_id)),
            _MODO_PROMPTING,
        ]
        for m in roster_activo()
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
    las dos columnas de continuidad 2025 (enmienda F8/RF19b)."""
    filas = []
    for fila in resumen:
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
    """Tabla 3: exactitud por categoría lingüística, una columna por modelo."""
    columnas_modelo = [c for c in df.columns if c not in ("categoria", "n")]
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
    """Tabla 4: taxonomía de 7 etiquetas de error, transpuesta (una columna por
    modelo, una fila por etiqueta más el total de incorrectas)."""
    modelos = list(df["modelo"])
    encabezado = ["Categoría de error"] + [_escapar(m) for m in modelos]
    fila_total = ["Total de respuestas incorrectas"] + [
        str(int(df.loc[df["modelo"] == m, "total_incorrectas"].iloc[0])) for m in modelos
    ]
    filas = [fila_total]
    for etiqueta in ETIQUETAS_ERROR:
        fila = [_escapar(ETIQUETAS_ERROR_DISPLAY[etiqueta])] + [
            str(int(df.loc[df["modelo"] == m, etiqueta].iloc[0])) for m in modelos
        ]
        filas.append(fila)
    spec = "l" + "r" * len(modelos)
    return _tabla(
        caption="Distribución de errores por categoría de la taxonomía",
        label="tab:taxonomia",
        spec=spec,
        encabezado=encabezado,
        filas=filas,
        ancho_completo=True,
    )


def tabla5_versiones() -> str:
    """Tabla 5: matriz de versiones de transformers (RF5), misma información que
    la matriz de docker/README.md."""
    filas = []
    for m in roster_activo():
        version_resuelta = _VERSION_RESUELTA[m.transformers_pin]
        necesario = "Sí" if m.motivo_pin else "Heredado"
        motivo = _escapar(m.motivo_pin) if m.motivo_pin else "—"
        filas.append([
            _escapar(m.nombre),
            _escapar(m.transformers_pin),
            version_resuelta,
            necesario,
            motivo,
        ])
    nota = (
        f"Grupo A (\\texttt{{{_escapar(BASELINE_TRANSFORMERS)}}}, resuelve "
        f"{_VERSION_RESUELTA[BASELINE_TRANSFORMERS]}) y grupo B "
        f"(\\texttt{{{_escapar(TRANSFORMERS_5X)}}}, resuelve "
        f"{_VERSION_RESUELTA[TRANSFORMERS_5X]}) son necesarios y mutuamente "
        "excluyentes sobre el roster: no existe una única versión mayor que "
        "sirva para las 12 filas activas."
    )
    return _tabla(
        caption="Pines de \\texttt{transformers} por modelo",
        label="tab:versiones",
        spec="llllp{4cm}",
        encabezado=["Modelo", "transformers\\_pin", "Versión resuelta", "¿Necesario?", "Motivo"],
        filas=filas,
        nota=nota,
        ancho_completo=True,
    )


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


if __name__ == "__main__":
    main()
