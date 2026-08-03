---
id: 11
title: Generador de fragmentos `.tex` de tablas
depends_on: [09]
files:
  - src/generate_tex_tables.py
  - tests/test_generate_tex_tables.py
  - paper/02_reescrito/tablas/
---

## Spec

Implementar **F8** / **RF13**: emitir las cinco tablas del paper reescrito como fragmentos `.tex` autónomos que `main.tex` incorpora con `\input`, generados desde los artefactos de datos. Ningún número del paper se escribe a mano — regenerar los fragmentos tras un cambio de datos no debe requerir tocar prosa.

Tablas: 1 (roster), 2 (resultados globales con estricta y laxa), 3 (por categoría lingüística), 4 (taxonomía de 7 etiquetas), 5 (matriz de versiones, que materializa **RF5** en el paper).

Este subtask crea `paper/02_reescrito/tablas/` pero **no** el resto del árbol LaTeX: eso es del subtask 14. Los fragmentos no llevan preámbulo ni `\documentclass`; se validan compilándolos contra un documento LNCS mínimo desechable.

## Implementation plan

### Tarea 1 — Escapado de LaTeX y helpers (TDD)

- [ ] Escribir el test que falla, `tests/test_generate_tex_tables.py` (primera parte):

```python
"""Tests del generador de fragmentos .tex de tablas (F8 del índice)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_tex_tables import (  # noqa: E402
    ARCHIVOS_TABLAS,
    _escapar,
    tabla1_modelos,
    tabla2_resultados,
    tabla3_por_categoria,
    tabla4_taxonomia,
    tabla5_versiones,
)
from models_2026 import (  # noqa: E402
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    TRANSFORMERS_5X,
    roster_activo,
)
from taxonomia_2026 import ETIQUETAS_ERROR  # noqa: E402


def test_escapa_los_caracteres_especiales_de_latex():
    assert _escapar("a_b") == r"a\_b"
    assert _escapar("100%") == r"100\%"
    assert _escapar("a&b") == r"a\&b"
    assert _escapar("x#1") == r"x\#1"
    assert _escapar("${}") == r"\$\{\}"


def test_escapa_los_guiones_bajos_de_los_nombres_de_etiqueta():
    assert _escapar("confusion_intencion") == r"confusion\_intencion"


def test_escapar_no_rompe_texto_limpio():
    assert _escapar("granite-4.0-h-1b") == "granite-4.0-h-1b"


def test_los_cinco_archivos_estan_declarados():
    assert sorted(ARCHIVOS_TABLAS) == [
        "tabla1_modelos.tex", "tabla2_resultados_globales.tex",
        "tabla3_por_categoria.tex", "tabla4_taxonomia.tex",
        "tabla5_versiones.tex",
    ]
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_generate_tex_tables.py`
- [ ] Implementar la base de `src/generate_tex_tables.py`:

```python
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
import sys
from pathlib import Path

import pandas as pd

from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo
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
           filas: list[list[str]], nota: str = "") -> str:
    """Arma un bloque table+tabular con booktabs, en el estilo LNCS."""
    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(encabezado) + r" \\",
        r"\midrule",
    ]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    if nota:
        lineas.append(f"\\\\[2pt]\\footnotesize{{{nota}}}")
    lineas.append(r"\end{table}")
    return "\n".join(lineas) + "\n"
```

- [ ] Correr y confirmar **verde** la primera parte: `pytest -q tests/test_generate_tex_tables.py`

### Tarea 2 — Las cinco tablas (TDD)

- [ ] Agregar a `tests/test_generate_tex_tables.py`:

```python
def _resumen():
    return [{
        "modelo": m.nombre, "hf_repo_id": m.hf_repo_id, "params_b": m.params_b,
        "tier": m.tier, "modo_prompting": "chat_template",
        "transformers_pin": m.transformers_pin, "n": 32, "json_valido_pct": 100.0,
        "exact_match_pct": 40.0 + i, "exact_match_laxo_pct": 45.0 + i,
        "avg_latencia_s": 10.0 + i, "acc_intent_pct": 80.0,
        "acc_dispositivo_pct": 70.0, "acc_ubicacion_pct": 75.0,
        "acc_valor_pct": 85.0, "acc_unidad_pct": 90.0,
    } for i, m in enumerate(roster_activo())]


def _es_bloque_table(tex: str) -> bool:
    return (tex.startswith(r"\begin{table}") and tex.rstrip().endswith(r"\end{table}")
            and r"\documentclass" not in tex and tex.count(r"\begin{tabular}") == 1)


def test_tabla1_lista_los_doce_modelos_del_roster_activo_con_su_label():
    tex = tabla1_modelos()
    assert _es_bloque_table(tex)
    assert r"\label{tab:modelos}" in tex
    for m in roster_activo():
        assert _escapar(m.nombre) in tex
    assert tex.count(r"\\") >= 12


def test_tabla1_no_lista_los_modelos_excluidos():
    tex = tabla1_modelos()
    excluidos = [m for m in MODELOS_2026 if not m.activo]
    assert len(excluidos) == 2
    for m in excluidos:
        assert _escapar(m.nombre) not in tex


def test_tabla2_trae_estricta_y_laxa():
    tex = tabla2_resultados(_resumen())
    assert _es_bloque_table(tex)
    assert r"\label{tab:globales}" in tex
    assert "Estricta" in tex and "Laxa" in tex
    assert "40.0" in tex and "45.0" in tex


def test_tabla3_una_fila_por_categoria():
    df = pd.DataFrame([
        {"categoria": "encendido_apagado_simple", "n": 8,
         roster_activo()[0].nombre: 50.0, roster_activo()[1].nombre: 62.5},
        {"categoria": "consulta_de_estado", "n": 3,
         roster_activo()[0].nombre: 33.3, roster_activo()[1].nombre: 66.7},
    ])
    tex = tabla3_por_categoria(df)
    assert _es_bloque_table(tex)
    assert r"\label{tab:categorias}" in tex
    assert CATEGORIAS_DISPLAY["encendido_apagado_simple"] in tex
    assert "(8)" in tex and "(3)" in tex          # el n va con la categoría
    assert "62.5" in tex


def test_tabla4_tiene_las_siete_etiquetas_con_nombre_legible():
    df = pd.DataFrame([
        {"modelo": roster_activo()[0].nombre, "total_incorrectas": 10,
         **{e: i for i, e in enumerate(ETIQUETAS_ERROR)}},
    ])
    tex = tabla4_taxonomia(df)
    assert _es_bloque_table(tex)
    assert r"\label{tab:taxonomia}" in tex
    for e in ETIQUETAS_ERROR:
        assert ETIQUETAS_ERROR_DISPLAY[e] in tex
    assert "Alucinación de valor/unidad" in tex
    assert "Valor numérico incorrecto" in tex


def test_tabla5_marca_cuales_pines_son_necesarios():
    tex = tabla5_versiones()
    assert _es_bloque_table(tex)
    assert r"\label{tab:versiones}" in tex
    for m in roster_activo():
        assert _escapar(m.nombre) in tex
    assert _escapar(BASELINE_TRANSFORMERS) in tex
    assert _escapar(TRANSFORMERS_5X) in tex
    assert "necesario" in tex.lower()


def test_tabla5_no_lista_los_modelos_excluidos():
    tex = tabla5_versiones()
    excluidos = [m for m in MODELOS_2026 if not m.activo]
    for m in excluidos:
        assert _escapar(m.nombre) not in tex


def test_ninguna_tabla_deja_guiones_bajos_sin_escapar():
    df_tax = pd.DataFrame([{"modelo": roster_activo()[0].nombre, "total_incorrectas": 1,
                            **{e: 0 for e in ETIQUETAS_ERROR}}])
    df_cat = pd.DataFrame([{"categoria": "encendido_apagado_simple", "n": 8,
                            roster_activo()[0].nombre: 50.0}])
    for tex in (tabla1_modelos(), tabla2_resultados(_resumen()),
                tabla3_por_categoria(df_cat), tabla4_taxonomia(df_tax),
                tabla5_versiones()):
        for i, ch in enumerate(tex):
            if ch == "_":
                assert tex[i - 1] == "\\", f"guion bajo sin escapar cerca de: {tex[i-30:i+10]!r}"
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_generate_tex_tables.py`
- [ ] Implementar las cinco funciones. Firmas: `tabla1_modelos() -> str`, `tabla2_resultados(resumen: list[dict]) -> str`, `tabla3_por_categoria(df: pd.DataFrame) -> str`, `tabla4_taxonomia(df: pd.DataFrame) -> str`, `tabla5_versiones() -> str`. Las tablas 1 y 5 se generan desde `models_2026.roster_activo()` (12 filas), **no** desde `MODELOS_2026` (14): los dos modelos excluidos no aparecen en ninguna tabla. Puntos obligatorios:
  - Tabla 1: columnas `Modelo | Parámetros | Tier | Familia | Prompting`; la familia sale del prefijo del `hf_repo_id` (`LiquidAI`, `ibm-granite`, `Qwen`, `HuggingFaceTB`, `allenai`); el prompting se toma de la tabla §2.1 del índice.
  - Tabla 2: `Modelo | Par. | JSON válido | Estricta | Laxa | Latencia (s)`, con los porcentajes con un decimal y la latencia con dos.
  - Tabla 3: primera columna `Categoría (n)` usando `CATEGORIAS_DISPLAY[cat]` y el `n` entre paréntesis; una columna por modelo.
  - Tabla 4: primera columna `Categoría de error` con `ETIQUETAS_ERROR_DISPLAY`; una fila `Total de respuestas incorrectas` primero y luego una fila por etiqueta; una columna por modelo (transpuesta respecto del CSV, igual que la Tabla 4 del paper original).
  - Tabla 5 (re-congelada, F8): `Modelo | transformers_pin | Versión resuelta | ¿Necesario? | Motivo`, generada desde `roster_activo()`. `¿Necesario?` es "Sí" si `motivo_pin` no está vacío, "Heredado" si sí; `Motivo` es `motivo_pin` o `—` si está vacío. "Versión resuelta" es `4.57.6` para el grupo A y `5.14.1` para el grupo B (constantes de módulo, no re-sondeadas). Nota al pie indicando cuáles son los dos grupos y sus constantes (`BASELINE_TRANSFORMERS`, `TRANSFORMERS_5X`). Debe ser la **misma información** que la matriz de `docker/README.md` (mismo conjunto de modelos, mismo pin, mismo motivo).
- [ ] Implementar el `main` que escribe los cinco archivos en `--salida-dir` y los lista por stdout.
- [ ] Correr y confirmar **verde**: `pytest -q tests/test_generate_tex_tables.py`

### Tarea 3 — Validar que los fragmentos compilan

> Exención de TDD: es una verificación de integración con el toolchain, no comportamiento nuevo.

- [ ] Generar los fragmentos: `python src/generate_tex_tables.py`
- [ ] Compilar un documento LNCS mínimo desechable que los incluya (en el scratch, no en el repo):

```bash
TMP=$(mktemp -d)
cp "LaTeX2e (1)/llncs.cls" "$TMP/"
cp paper/02_reescrito/tablas/*.tex "$TMP/"
cat > "$TMP/prueba.tex" <<'EOF'
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-tabla,es-noquoting]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\begin{document}
\title{prueba}\author{a}\institute{b}\maketitle
\input{tabla1_modelos}
\input{tabla2_resultados_globales}
\input{tabla3_por_categoria}
\input{tabla4_taxonomia}
\input{tabla5_versiones}
\end{document}
EOF
(cd "$TMP" && latexmk -pdf -interaction=nonstopmode -halt-on-error prueba.tex) \
  && echo "las 5 tablas compilan" || echo "FALLA de compilacion"
```

- [ ] Si alguna tabla se sale del ancho de página, envolverla en `\resizebox{\textwidth}{!}{...}` dentro de `_tabla` (las de 12 columnas — Tablas 3 y 4 — son las candidatas) y recompilar.

### Tarea 4 — commit

- [ ] `pytest -q`
- [ ] `git add src/generate_tex_tables.py tests/test_generate_tex_tables.py paper/02_reescrito/tablas/`
- [ ] `git commit -m "feat(2026): generador de las 5 tablas .tex del paper desde los datos"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Los cinco fragmentos existen y son bloques table autónomos
python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from generate_tex_tables import ARCHIVOS_TABLAS
for n in ARCHIVOS_TABLAS:
    t = (Path('paper/02_reescrito/tablas')/n).read_text(encoding='utf-8')
    assert t.startswith('\\\\begin{table}'), n
    assert t.rstrip().endswith('\\\\end{table}'), n
    assert '\\\\documentclass' not in t, n
    assert '\\\\label{tab:' in t, n
print('5 fragmentos OK')
"

# 3. Los números del .tex coinciden con los del JSON
python -c "
import json
from pathlib import Path
r = json.load(open('data/2026/resumen_2026.json', encoding='utf-8'))
t = Path('paper/02_reescrito/tablas/tabla2_resultados_globales.tex').read_text(encoding='utf-8')
for f in r:
    assert f'{f[\"exact_match_pct\"]:.1f}' in t, f['modelo']
    assert f'{f[\"exact_match_laxo_pct\"]:.1f}' in t, f['modelo']
print('tabla 2 coincide con el JSON')
"

# 4. Regenerar no produce diff (los datos son la única fuente)
python src/generate_tex_tables.py >/dev/null
git diff --exit-code -- paper/02_reescrito/tablas/ && echo "regeneracion idempotente OK"

# 5. Compilación de las 5 tablas (ver Tarea 3)

# 6. RF5: los pines necesarios aparecen en la tabla 5, y los excluidos no aparecen
python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from models_2026 import MODELOS_2026, TRANSFORMERS_5X, roster_activo
t = Path('paper/02_reescrito/tablas/tabla5_versiones.tex').read_text(encoding='utf-8')
for m in roster_activo():
    assert m.nombre.replace('.', chr(92)+'.') in t or m.nombre in t, m.nombre
    if m.motivo_pin:
        assert m.transformers_pin in t, f'{m.nombre}: falta su pin en la tabla 5'
for m in MODELOS_2026:
    if not m.activo:
        assert m.nombre not in t, f'{m.nombre}: excluido pero aparece en la tabla 5'
print('tabla 5 refleja los pines necesarios y omite los excluidos')
"
```

## Acceptance criteria

- **Dado** `_escapar`, **entonces** convierte `_ % & # $ { } ~ ^ \` a sus formas LaTeX seguras y deja intacto el texto limpio (`granite-4.0-h-1b`).
- **Dado** cualquiera de los cinco fragmentos, **entonces** empieza con `\begin{table}`, termina con `\end{table}`, contiene exactamente un `tabular`, no contiene `\documentclass` ni preámbulo, y declara su `\label{tab:...}` congelado en F8.
- **Dado** la Tabla 1, **entonces** lista los 12 modelos del roster activo (`roster_activo()`) con parámetros, tier, familia y modo de prompting; los dos modelos excluidos no aparecen.
- **Dado** la Tabla 2, **entonces** tiene columnas para JSON válido, exactitud **estricta**, exactitud **laxa** y latencia, y sus valores coinciden dígito a dígito con `data/2026/resumen_2026.json`.
- **Dado** la Tabla 3, **entonces** tiene una fila por categoría lingüística presente, con el nombre legible de `CATEGORIAS_DISPLAY` y el `n` entre paréntesis, y una columna por modelo.
- **Dado** la Tabla 4, **entonces** tiene una fila `Total de respuestas incorrectas` y **siete** filas de etiquetas con sus nombres legibles, incluyendo `Alucinación de valor/unidad` y `Valor numérico incorrecto` como filas **separadas**.
- **Dado** la Tabla 5, **entonces** lista los 12 modelos del roster activo con su `transformers_pin`, su versión resuelta (`4.57.6` grupo A / `5.14.1` grupo B), si el pin es **necesario** o **heredado**, y su motivo (`—` cuando es heredado), con una nota al pie que declara los dos grupos; es la misma información que la matriz de `docker/README.md`.
- **Dado** cualquier fragmento, **entonces** no contiene ningún `_` sin escapar (todos precedidos por `\`).
- **Dado** un documento LNCS mínimo que hace `\input` de los cinco fragmentos, **entonces** `latexmk -pdf -halt-on-error` compila con código 0.
- **Dado** que se regenera sin cambiar los datos, **entonces** `git diff -- paper/02_reescrito/tablas/` sale vacío.
