---
id: 14
title: "Paper reescrito: andamiaje y secciones 1–3"
depends_on: [10, 11, 12, 13]
files:
  - paper/02_reescrito/main.tex
  - paper/02_reescrito/refs.bib
  - paper/02_reescrito/llncs.cls
  - paper/02_reescrito/splncs03.bst
  - paper/02_reescrito/figuras/
  - paper/02_reescrito/secciones/00_abstract.tex
  - paper/02_reescrito/secciones/01_introduccion.tex
  - paper/02_reescrito/secciones/02_trabajos_relacionados.tex
  - paper/02_reescrito/secciones/03_metodologia.tex
  # stubs creados acá para que main.tex compile; los completa el subtask 15
  - paper/02_reescrito/secciones/04_resultados.tex
  - paper/02_reescrito/secciones/05_discusion.tex
  - paper/02_reescrito/secciones/06_amenazas.tex
  - paper/02_reescrito/secciones/07_conclusiones.tex
  - paper/02_reescrito/secciones/08_declaracion_ia.tex
---

## Spec

Primera mitad del paper reescrito (**RF14**): el andamiaje LNCS anónimo más las secciones que **no** dependen de los resultados numéricos — abstract, introducción, trabajos relacionados y metodología.

La metodología es la sección que más cambia respecto del original, porque cambió el experimento: 14 modelos en vez de 4 (Tabla 1 generada), prompt con cláusula taxativa (**RF1**), ruta de raw completion para los dos modelos base (**RF2**), aislamiento por contenedor y matriz de versiones (**RF4**, **RF5**, Tabla 5), y sobre todo el **pipeline automático de dos etapas** que reemplaza el etiquetado manual (**RF6**–**RF9**), con sus dos métricas (**RF10**). También se documenta, para envío ciego (**RF18**), que dos de los catorce modelos son *gated* y qué implica eso para la reproducibilidad por terceros.

Restricción dura desde el primer commit: **envío ciego** (**RF15**). El bloque de autor se escribe anonimizado y el verificador del subtask 12 debe salir en 0 ya en este subtask, no recién en el 15.

Las secciones 4–7 son del subtask 15.

## Implementation plan

### Tarea 1 — Andamiaje anónimo (TDD sobre el anonimato)

- [ ] Crear el árbol y copiar clase, estilo y figuras:

```bash
mkdir -p paper/02_reescrito/secciones paper/02_reescrito/figuras
cp "LaTeX2e (1)/llncs.cls"    paper/02_reescrito/
cp "LaTeX2e (1)/splncs03.bst" paper/02_reescrito/
cp figures/2026/fig1_exactitud_latencia_2026.png paper/02_reescrito/figuras/
cp figures/2026/fig2_exactitud_por_campo_2026.png paper/02_reescrito/figuras/
```

- [ ] Escribir `paper/02_reescrito/main.tex` con el preámbulo congelado de **F9** y el bloque de autor **anonimizado**. Los metadatos del PDF se fuerzan vacíos con `hyperref` (que `llncs` ya carga) — es la fuga que el `.tex` por sí solo no evita:

```latex
% Paper para CACIC 2026 --- ENVÍO CIEGO.
% No agregar autores, afiliaciones, agradecimientos, financiamiento, URL del
% repositorio ni ORCID. Verificar con:
%     python scripts/check_anonimato.py paper/02_reescrito
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-tabla,es-noquoting]{babel}
\usepackage{graphicx}
\usepackage{booktabs}

\hypersetup{pdfauthor={},pdfsubject={},pdfkeywords={}}

\begin{document}

\title{Modelos de Lenguaje Pequeños para la Interpretación de Comandos de
Domótica en Español: Evaluación Automática de Catorce Modelos Sub-2B}
\author{Anonimizado por revisión ciega}
\institute{Anonimizado por revisión ciega}
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

- [ ] Crear los cinco archivos de sección que aún no se escriben en este subtask, como stubs mínimos compilables, para que `main.tex` compile desde ya (el subtask 15 los reemplaza):

```bash
for s in 04_resultados 05_discusion 06_amenazas 07_conclusiones 08_declaracion_ia; do
  printf '%% Se completa en el subtask 15.\n' > "paper/02_reescrito/secciones/$s.tex"
done
```

- [ ] Copiar `refs.bib` desde la transcripción y verificar que compila la bibliografía:
  `cp paper/01_original/refs.bib paper/02_reescrito/refs.bib`

### Tarea 2 — Abstract, introducción y trabajos relacionados

- [ ] `00_abstract.tex` — `\begin{abstract}...\end{abstract}` + `\keywords{...}`, **solo en español** (A8: no va abstract en inglés). Debe declarar, con estas cuatro piezas: (a) que se evalúan **14** SLM abiertos de menos de 2B en interpretación de comandos de domótica en español rioplatense; (b) que la verificación es **totalmente automática** en dos etapas, coincidencia textual más un juez LLM con categorías cerradas, sin etiquetado manual; (c) que se reportan **dos** métricas, exactitud estricta y laxa, y que la brecha entre ambas cuantifica cuánto penaliza la coincidencia exacta a respuestas semánticamente correctas; (d) que todo corre en CPU con 2 núcleos y 8 GB, un contenedor por modelo. Sin números concretos: los resultados van en §4.
- [ ] `01_introduccion.tex` — mantener la motivación del original (SLM locales para domótica, privacidad, hardware modesto) y agregar el aporte nuevo: el original clasificó a mano sus 73 respuestas incorrectas, lo que no escala a 14 modelos ni es reproducible; este trabajo automatiza ese juicio. Cerrar con las contribuciones enumeradas: roster de 14 modelos 2026, pipeline de verificación automática en dos etapas, doble métrica estricta/laxa, y protocolo reproducible con aislamiento por contenedor.
- [ ] `02_trabajos_relacionados.tex` — reutilizar el texto del original (las 9 referencias siguen siendo válidas) y **agregar un párrafo nuevo sobre LLM-as-judge**, que es la técnica central que se incorpora: mencionar que la evaluación con modelos como jueces se usa cuando la coincidencia exacta subestima el desempeño, y que su limitación conocida es el sesgo de auto-favorecimiento — lo que enlaza con §6. Agregar a `refs.bib` **al menos una** referencia de LLM-as-judge y una de los modelos nuevos que se evalúan (familias LFM2.5 / Granite 4.0 / Qwen3.5 / OLMo-2). Las referencias nuevas **no deben deanonimizar**: nada de autocitas.

### Tarea 3 — Metodología (la sección que más cambia)

- [ ] `03_metodologia.tex` con `\section{Metodología}` y estas subsecciones:

**3.1 Modelos evaluados** — `\input{tablas/tabla1_modelos}`. Explicar el criterio de selección: modelos abiertos de dos tiers (sub-1B y 1–2B); indicar explícitamente que dos de los catorce son *gated* —requieren aceptar una licencia de uso y disponer de un token propio para descargarse— y que se incluyeron porque no representan una barrera insalvable para este trabajo, a diferencia de otros modelos descartados por estar superados por alternativas más recientes de la misma familia; remitir a 3.6 para la nota de reproducibilidad sobre los modelos *gated*. Señalar también que `SmolLM2` aparece en sus dos tamaños como continuidad con el estudio anterior. Declarar que **dos** de los catorce (`LFM2.5-230M` y `LFM2.5-350M`) son modelos **base**, sin plantilla de chat.

**3.2 Dataset de comandos** — sin cambios de fondo respecto del original: los mismos 32 comandos, el mismo esquema de 5 campos. Aclarar explícitamente que el dataset **no se modificó** para este estudio y que las categorías lingüísticas de §4.3 ya **no** se asignan a mano (ver 3.5).

**3.3 Protocolo de prompting y esquema de salida** — el cambio de **RF1**. Transcribir la cláusula taxativa en un `verbatim` o `quote` y explicar la decisión: los campos con valores entre paréntesis son categóricos cerrados y esos valores son los únicos outputs aceptados **textualmente**; usar un sinónimo es una violación de formato. Explicar el corolario metodológico: esto endurece deliberadamente el protocolo respecto del estudio anterior, así que la exactitud estricta **no es comparable** con la del paper original. Documentar también **RF2**: los dos modelos base se prompean por *raw completion* con una plantilla de texto plano, decidido en tiempo de ejecución por capacidad del tokenizer y no por identificador.

**3.4 Métricas** — las cinco del original más la sexta nueva. Definir con precisión:
- *exactitud estricta*: los 5 campos coinciden textualmente con el ground truth;
- *exactitud laxa (semántica)*: estricta, **o** el juez etiquetó la respuesta como `sin_error_semantico` o `uso_de_sinonimos`.
Declarar que la estricta es la métrica **primaria y conforme al protocolo**, y la laxa una cota superior semántica; y que la **brecha** entre ambas es un resultado en sí mismo, no una corrección.

**3.5 Verificación automática en dos etapas** — subsección **nueva**, el corazón del aporte. Describir:
1. Etapa 1: comparación textual campo a campo, determinista.
2. Selección del juez: el modelo con mayor exactitud estricta en la etapa 1, con desempate por mayor cantidad de parámetros. Aclarar que la reproducibilidad se garantiza por **decodificación greedy (temperatura 0)** en ambas etapas, no por fijar el identificador del juez.
3. Etapa 2, trabajo A: cada respuesta incorrecta se clasifica en un subconjunto no vacío de **siete** etiquetas cerradas — las cinco de la taxonomía publicada más `sin_error_semantico` y `uso_de_sinonimos`. Salida cerrada: si no parsea contra el vocabulario, se reintenta una vez y luego se cae a una etiqueta de respaldo determinista, y se reporta cuántas veces ocurrió.
4. Etapa 2, trabajo B: cada uno de los 32 comandos se clasifica en exactamente una de las **seis** categorías lingüísticas, reemplazando el etiquetado manual de la Tabla 3.
Cerrar señalando que el juez evalúa también sus propias salidas y que eso se discute en §6.

**3.6 Entorno de ejecución y reproducibilidad** — **RF4** y **RF5**. Hardware: CPU, 2 núcleos, 8 GB, sin GPU, igual que el estudio original para que las latencias sean comparables. Aislamiento: una imagen de contenedor por modelo, ejecutadas de a una, compartiendo la caché de pesos. Y el punto que el usuario pidió explícitamente: `\input{tablas/tabla5_versiones}` con la matriz de versiones, más un párrafo que explique **por qué** difieren (arquitecturas recientes que exigen versiones distintas de la librería de inferencia), **qué modelo forzó cada divergencia**, y que esto es un confusor conocido para la comparación de latencia que se retoma en §6.

- [ ] Si en el subtask 04 **ningún** pin resultó divergente, decirlo explícitamente ("todos los modelos corrieron con la misma versión de la librería de inferencia") en vez de omitir la subsección: la ausencia de divergencia también es información de reproducibilidad.
- [ ] Cerrar 3.6 con el párrafo de **RF18** (reproducibilidad y modelos *gated*, redactado para envío ciego, sin datos identificatorios y sin mencionar el token concreto): dos de los catorce modelos (`gemma-3-270m-it` y `Llama-3.2-1B-Instruct`) son *gated*; descargarlos exige aceptar su licencia respectiva (Gemma Terms of Use y Llama 3.2 Community License) y disponer de un token propio de HuggingFace (`$HF_TOKEN`, nunca su valor concreto). Esto limita la reproducibilidad del barrido completo por parte de terceros que no dispongan de cuenta propia con esas licencias aceptadas, a diferencia de los otros doce modelos, que se descargan sin restricciones.

### Tarea 4 — Compilar, verificar anonimato y commitear

- [ ] Generar las tablas si hiciera falta: `python src/generate_tex_tables.py`
- [ ] Compilar: `(cd paper/02_reescrito && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)`
- [ ] Verificar anonimato **ya en este subtask**: `python scripts/check_anonimato.py paper/02_reescrito` → código 0
- [ ] `pytest -q`
- [ ] `git add paper/02_reescrito/`
- [ ] `git commit -m "docs(paper): andamiaje anonimo y secciones 1-3 del paper reescrito"`

## Verify

```bash
# 1. Layout de F9 completo
ls paper/02_reescrito/{main.tex,refs.bib,llncs.cls,splncs03.bst}
ls -1 paper/02_reescrito/secciones/*.tex | wc -l     # -> 9 (4 escritas + 5 stubs)
ls -1 paper/02_reescrito/tablas/*.tex | wc -l        # -> 5
ls paper/02_reescrito/figuras/*.png

# 2. Compila y produce PDF
(cd paper/02_reescrito && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex) \
  && test -f paper/02_reescrito/main.pdf && echo "compila OK"

# 3. ENVÍO CIEGO: el verificador sale en 0, incluido el PDF ya construido
python scripts/check_anonimato.py paper/02_reescrito
test $? -eq 0 && echo "anonimato OK"

# 4. La metodología incorpora las tablas generadas, no números a mano
grep -q 'input{tablas/tabla1_modelos}'   paper/02_reescrito/secciones/03_metodologia.tex
grep -q 'input{tablas/tabla5_versiones}' paper/02_reescrito/secciones/03_metodologia.tex \
  && echo "tablas 1 y 5 enlazadas"

# 5. La metodología cubre los requisitos que le tocan
python -c "
from pathlib import Path
t = Path('paper/02_reescrito/secciones/03_metodologia.tex').read_text(encoding='utf-8').lower()
faltan = [k for k in ('taxativ','sinónimo','raw completion','estricta','laxa',
                      'greedy','contenedor','transformers','juez','categor','gated')
          if k not in t]
assert not faltan, faltan
print('metodologia cubre: prompt taxativo, raw completion, doble metrica, juez, docker, versiones, gated')
"

# 6. Abstract solo en español
python -c "
from pathlib import Path
t = Path('paper/02_reescrito/secciones/00_abstract.tex').read_text(encoding='utf-8')
assert t.count('begin{abstract}') == 1, 'hay mas de un abstract'
print('abstract unico (español) OK')
"

# 7. Nada fuera de alcance
git diff --name-only main | grep -v '^paper/02_reescrito/' \
  && echo "OJO: cambios fuera del alcance" || echo "alcance respetado"
```

## Acceptance criteria

- **Dado** `paper/02_reescrito/`, **entonces** tiene el layout de F9: `main.tex`, `refs.bib`, `llncs.cls`, `splncs03.bst`, 9 archivos en `secciones/`, 5 en `tablas/` y las 2 figuras en `figuras/`.
- **Dado** `main.tex`, **entonces** usa el preámbulo congelado, hace `\input` de las nueve secciones en orden, y su bloque de autor e institución está **anonimizado**, con `\hypersetup` forzando `pdfauthor`, `pdfsubject` y `pdfkeywords` vacíos.
- **Dado** `latexmk -pdf -halt-on-error main.tex`, **entonces** termina con código 0 y produce `main.pdf`, aun con las cinco secciones que todavía son stubs.
- **Dado** `python scripts/check_anonimato.py paper/02_reescrito`, **entonces** sale con **código 0**, revisando tanto los `.tex` como el texto y los metadatos del `main.pdf` construido.
- **Dado** el abstract, **entonces** está solo en español, en un único bloque `abstract`, y menciona los 14 modelos, la verificación automática en dos etapas y las dos métricas, sin dar cifras de resultados.
- **Dado** `02_trabajos_relacionados.tex`, **entonces** incluye un párrafo sobre LLM-as-judge con al menos una referencia nueva en `refs.bib`, y ninguna referencia nueva deanonimiza a los autores.
- **Dado** `03_metodologia.tex`, **entonces** hace `\input` de `tablas/tabla1_modelos` y `tablas/tabla5_versiones` (ningún número escrito a mano) y cubre explícitamente: la cláusula taxativa transcripta y su consecuencia sobre la comparabilidad con el estudio anterior; el prompting por raw completion de los dos modelos base decidido por capacidad; la definición precisa de exactitud estricta y laxa; el pipeline de dos etapas con la regla de selección del juez y la decodificación greedy; las siete etiquetas y las seis categorías cerradas con su política de reintento y respaldo; y el entorno de 2 núcleos / 8 GB con un contenedor por modelo.
- **Dado** §3.6, **entonces** declara qué modelo forzó cada divergencia de versión y por qué, o bien afirma explícitamente que no hubo ninguna divergencia; y declara, sin datos identificatorios ni mención del valor del token, que dos de los catorce modelos son *gated* (nombrando la licencia de cada uno) y que esto limita la reproducibilidad del barrido por terceros.
- **Dado** `git diff --name-only main`, **entonces** los únicos cambios están bajo `paper/02_reescrito/`.
