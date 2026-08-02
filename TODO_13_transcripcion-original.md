---
id: 13
title: Transcripción fiel del `.docx` a `paper/01_original/`
depends_on: []
files:
  - paper/01_original/main.tex
  - paper/01_original/refs.bib
  - paper/01_original/llncs.cls
  - paper/01_original/splncs03.bst
  - paper/01_original/secciones/
---

## Spec

Primera mitad de **RF14**: transcribir `paper_cacic_LNCS_word.docx` a un árbol LaTeX LNCS **fiel**, con `main.tex` y un `.tex` por sección, según el layout congelado en **F9**.

"Fiel" significa: mismo texto, mismas secciones, mismas tablas con los mismos números, mismas referencias, mismos placeholders de autor. **No se corrige, no se mejora, no se actualiza nada** — este árbol es la línea de base de trazabilidad contra la que se lee el reescrito. Los errores del original (por ejemplo, que §4.4 diga "cuatro categorías" mientras la Tabla 4 lista cinco) se transcriben **tal cual**.

El `.docx` **no se commitea** (**A12**): queda en `.gitignore`, legible en disco como fuente.

Sin dependencias en el DAG: puede correr en paralelo con todo lo demás.

## Implementation plan

### Tarea 1 — Andamiaje LaTeX

> Exención de TDD: creación de estructura y configuración.

- [ ] Crear el árbol y copiar la clase y el estilo bibliográfico desde la plantilla oficial descargada por el usuario:

```bash
mkdir -p paper/01_original/secciones
cp "LaTeX2e (1)/llncs.cls"    paper/01_original/
cp "LaTeX2e (1)/splncs03.bst" paper/01_original/
```

> `.gitignore` **no se toca acá**: su único dueño en todo el DAG es el subtask 01,
> que ya escribe las entradas del `.docx`, de la plantilla descargada y de los
> auxiliares de LaTeX. Este subtask solo debe asegurarse de que su `git add` esté
> acotado a `paper/01_original/`, de modo que el `.docx` no se cuele aunque el
> subtask 01 todavía no haya corrido.

- [ ] Crear `paper/01_original/main.tex` con el preámbulo congelado de **F9**:

```latex
% Transcripción fiel de paper_cacic_LNCS_word.docx.
% NO corregir ni actualizar nada: este árbol es la línea de base de
% trazabilidad contra la que se compara paper/02_reescrito/.
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-tabla,es-noquoting]{babel}
\usepackage{graphicx}
\usepackage{booktabs}

\begin{document}

\title{Modelos de Lenguaje Pequeños para la Interpretación de Comandos de
Domótica en Español: Un Estudio Comparativo}
\author{[Nombre y Apellido del autor/a]}
\institute{[Afiliación / Institución]\\ \email{[correo@institucion.edu.ar]}}
\maketitle

\input{secciones/00_abstract}
\input{secciones/01_introduccion}
\input{secciones/02_trabajos_relacionados}
\input{secciones/03_metodologia}
\input{secciones/04_resultados}
\input{secciones/05_discusion}
\input{secciones/06_amenazas}
\input{secciones/07_conclusiones}
\input{secciones/08_declaracion_ia}

\bibliographystyle{splncs03}
\bibliography{refs}

\end{document}
```

### Tarea 2 — Extracción del contenido del `.docx`

El TODO original pide explícitamente hacer esto con un sub-agente. Delegar a un sub-agente `general-purpose` (modelo `sonnet`) con este encargo, y **no** volcar el `.docx` completo en el contexto principal:

> Extraé el contenido textual completo de `paper_cacic_LNCS_word.docx` (un `.docx` es un zip de XML; alcanza con `zipfile` + `xml.etree` de la stdlib, escribiendo el script auxiliar en el scratch, no en el repo). Devolvé, por cada sección del documento y en orden: el encabezado exacto, el texto completo de sus párrafos, y las tablas con sus captions y todas sus celdas. Reproducí el texto **verbatim**, sin corregir ortografía, redacción ni inconsistencias. Listá también las 9 referencias completas y los captions de las 2 figuras.

- [ ] Con esa extracción, escribir los nueve archivos de `paper/01_original/secciones/`, con estos contenidos y estructura de referencia (**verificados contra el `.docx`**):

| archivo | contenido |
|---|---|
| `00_abstract.tex` | `\begin{abstract}...\end{abstract}` + `\keywords{...}`. Solo español, sin versión en inglés. |
| `01_introduccion.tex` | `\section{Introducción}`, ~320 palabras |
| `02_trabajos_relacionados.tex` | `\section{Trabajos relacionados}`, ~559 palabras |
| `03_metodologia.tex` | `\section{Metodología}` + 5 subsecciones: `Modelos evaluados` (con la Tabla 1), `Dataset de comandos`, `Protocolo de prompting y esquema de salida`, `Métricas`, `Prompt utilizado` |
| `04_resultados.tex` | `\section{Experimentación y resultados}` + 4 subsecciones: `Resultados globales` (Tabla 2, Fig. 1), `Exactitud por campo del esquema` (Fig. 2), `Exactitud por categoría de comando` (Tabla 3), `Análisis cualitativo de errores` (Tabla 4) |
| `05_discusion.tex` | `\section{Discusión}`, ~307 palabras |
| `06_amenazas.tex` | `\section{Amenazas a la Validez}` con los cuatro subtítulos en negrita del original: validez de constructo, interna, externa y de conclusión |
| `07_conclusiones.tex` | `\section{Conclusiones y trabajo futuro}` |
| `08_declaracion_ia.tex` | `\section*{Declaración sobre el uso de IA}` |

- [ ] Transcribir las **cuatro tablas** con sus valores exactos. Datos de control que deben aparecer literalmente:
  - **Tabla 1** (`Modelos evaluados en este estudio.`): SmolLM2-360M-Instruct 0.36 B, Qwen2.5-0.5B-Instruct 0.49 B, Qwen2.5-1.5B-Instruct 1.54 B, SmolLM2-1.7B-Instruct 1.71 B.
  - **Tabla 2** (`Resultados globales por modelo (n=32 comandos).`): exactitudes 18.8 / 43.8 / 50.0 / 59.4 %, JSON válido 100 % en los cuatro, latencias 15.6 / 14.7 / 43.2 / 61.9 s.
  - **Tabla 3** (`Coincidencia exacta por categoría lingüística de comando y modelo.`): seis filas con n = 8, 8, 6, 4, 3, 3.
  - **Tabla 4** (`Taxonomía de errores por modelo...`): totales de incorrectas 26 / 18 / 16 / 13 y **cinco** categorías de error.
- [ ] Insertar las dos figuras referenciando los PNG publicados, con sus captions originales:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{../../figures/fig1_exactitud_latencia.png}
\caption{Coincidencia exacta (izquierda) y latencia promedio de inferencia en
CPU (derecha), por modelo.}
\label{fig:orig-globales}
\end{figure}
```

- [ ] Crear `paper/01_original/refs.bib` con las **9** referencias en formato BibTeX, y reemplazar las citas del texto por `\cite{...}`. Las tres primeras, verbatim del original:
  - Gupta, A., Thomas, B., Asnani, H., et al.: *Small Language Models (SLMs) Can Still Pack a Punch: A Survey.* arXiv:2501.05465 (2025)
  - Weld, H., Huang, X., Long, S., Poon, J., Han, S.C.: *A Survey of Joint Intent Detection and Slot Filling Models in Natural Language Understanding.* ACM Computing Surveys 55(8) (2023)
  - Bouraoui, J.L., et al.: *Towards a French Smart-Home Voice Command Corpus: Design and NLU Experiments.* In: Text, Speech, and Dialogue (TSD 2018). LNCS, Springer, Heidelberg

### Tarea 3 — Compilar y cotejar

- [ ] Compilar:

```bash
cd paper/01_original && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

- [ ] Resolver los problemas típicos de este combo antes de dar por buena la compilación: `°C` y `ñ` bajo `utf8`+`T1`; `%` literal dentro de las tablas (debe ir `\%`); guiones bajos en nombres de modelo; `babel` español y las comillas (por eso `es-noquoting` en el preámbulo).
- [ ] Cotejar el PDF resultante contra el `.docx`: mismas secciones y en el mismo orden, mismos números en las cuatro tablas, mismas dos figuras con sus captions, 9 referencias.
- [ ] Contar palabras por sección y comparar contra el original (240 / 320 / 559 / 95 / 86 / 100 / 59 / 179 / 157 / 105 / 125 / 396 / 307 / 267 / 188 / 126). Una desviación grande indica texto omitido:

```bash
python -c "
from pathlib import Path
import re
for p in sorted(Path('paper/01_original/secciones').glob('*.tex')):
    texto = re.sub(r'\\\\[a-zA-Z]+\\*?(\\[[^]]*\\])?(\\{[^}]*\\})?', ' ', p.read_text(encoding='utf-8'))
    print(f'{p.name:32s} {len(texto.split()):5d} palabras')
"
```

### Tarea 4 — commit

- [ ] Confirmar que el `.docx` no se va a commitear: `git status --short paper_cacic_LNCS_word.docx` no debe quedar en el índice tras el `git add` acotado de abajo
- [ ] `pytest -q`
- [ ] `git add paper/01_original/`
- [ ] `git commit -m "docs(paper): transcripcion fiel del docx original a LaTeX LNCS por secciones"`

## Verify

```bash
# 1. El árbol tiene el layout congelado en F9
ls paper/01_original/{main.tex,refs.bib,llncs.cls,splncs03.bst}
ls -1 paper/01_original/secciones/*.tex | wc -l     # -> 9

# 2. Compila sin errores
(cd paper/01_original && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex) \
  && echo "01_original compila"
test -f paper/01_original/main.pdf

# 3. main.tex hace \input de las nueve secciones y nada más
python -c "
import re
from pathlib import Path
t = Path('paper/01_original/main.tex').read_text(encoding='utf-8')
ins = re.findall(r'\\\\input\{secciones/([^}]+)\}', t)
assert len(ins) == 9, ins
for n in ins:
    assert Path(f'paper/01_original/secciones/{n}.tex').exists(), n
print('9 secciones enlazadas OK')
"

# 4. Los números publicados están presentes (transcripción fiel, no reinterpretada)
python -c "
from pathlib import Path
tex = ''.join(p.read_text(encoding='utf-8') for p in Path('paper/01_original/secciones').glob('*.tex'))
for v in ('18.8','43.8','50.0','59.4','15.6','14.7','43.2','61.9','26','18','16','13'):
    assert v in tex, v
for m in ('SmolLM2-360M','Qwen2.5-0.5B','Qwen2.5-1.5B','SmolLM2-1.7B'):
    assert m in tex, m
print('numeros y modelos del original presentes')
"

# 5. Nueve referencias
grep -c '^@' paper/01_original/refs.bib      # -> 9

# 6. El docx no está trackeado
git ls-files | grep -i docx && echo "FALLA: docx trackeado" || echo "docx sin trackear OK"

# 7. Los archivos congelados y los del reescrito no fueron tocados
git diff --name-only main | grep -v '^paper/01_original/' \
  && echo "OJO: hay cambios fuera del alcance" || echo "alcance respetado"
```

## Acceptance criteria

- **Dado** `paper/01_original/`, **entonces** contiene `main.tex`, `refs.bib`, `llncs.cls`, `splncs03.bst` y exactamente **9** archivos en `secciones/`, con los nombres congelados en F9.
- **Dado** `main.tex`, **entonces** usa el preámbulo congelado (`\documentclass[runningheads]{llncs}` con `fontenc`, `inputenc`, `babel` español, `graphicx`, `booktabs`), hace `\input` de las nueve secciones en orden, y declara `\bibliographystyle{splncs03}`.
- **Dado** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` en `paper/01_original/`, **entonces** termina con código 0 y produce `main.pdf`.
- **Dado** el PDF generado, **entonces** reproduce las mismas secciones y subsecciones del `.docx`, en el mismo orden, incluidas las cuatro tablas y las dos figuras con sus captions originales.
- **Dado** las cuatro tablas, **entonces** sus valores coinciden **exactamente** con los del `.docx`: exactitudes 18.8/43.8/50.0/59.4 %, JSON válido 100 %, latencias 15.6/14.7/43.2/61.9 s, totales de incorrectas 26/18/16/13, y la Tabla 4 con sus **cinco** categorías de error.
- **Dado** el texto transcripto, **entonces** conserva las inconsistencias del original sin corregirlas (§4.4 sigue diciendo "cuatro categorías" aunque la Tabla 4 liste cinco), y conserva los placeholders `[Nombre y Apellido del autor/a]` y `[Afiliación / Institución]`.
- **Dado** el abstract, **entonces** está solo en español, sin versión en inglés.
- **Dado** `refs.bib`, **entonces** tiene 9 entradas y todas las citas del texto resuelven (sin `?` en el PDF).
- **Dado** `git ls-files`, **entonces** `paper_cacic_LNCS_word.docx` **no** aparece: sigue sin trackear (su entrada en `.gitignore` la escribe el subtask 01, que es su único dueño).
- **Dado** `git diff --name-only main`, **entonces** los únicos cambios están bajo `paper/01_original/`.
