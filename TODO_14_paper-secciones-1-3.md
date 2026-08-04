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

La metodología es la sección que más cambia respecto del original, porque cambió el experimento: **12 modelos del roster activo** en vez de 4 (Tabla 1 generada desde `roster_activo()`), prompt con cláusula taxativa (**RF1**), detección de modo de prompting por capacidad (**RF2** — ningún modelo del roster activo ejercita *raw completion*: los 12 resuelven a `chat_template`), aislamiento por contenedor y matriz de versiones **por grupo** (**RF4**, **RF5**, Tabla 5: dos grupos de versión de `transformers` mutuamente excluyentes sobre el roster, 9 en el grupo A y 3 en el grupo B), y sobre todo el **pipeline automático de dos etapas** que reemplaza el etiquetado manual (**RF6**–**RF9**), con sus dos métricas (**RF10**). También se documenta, para envío ciego (**RF18**), que **3 de los 15 modelos del registro `MODELOS_2026` quedaron fuera del barrido**, por **dos causas distintas**: 2 (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) por acceso de descarga no otorgado, y 1 (`Qwen3.5-2B`) por compatibilidad de versión no verificada — ninguna de las tres exclusiones es una decisión metodológica —, y que los 12 evaluados son todos de repositorios públicos. Además (**RF19**), el roster reincorpora deliberadamente a `Qwen2.5-1.5B-Instruct` para sostener la continuidad con la Tabla 2 del paper original: los modelos presentes en ambos estudios pasan de 2 a 3, cubriendo ahora los dos tiers y dos familias distintas, lo que habilita una comparación explícita entre la exactitud estricta y la latencia nuevas y las publicadas — una validación cruzada del nuevo arnés de evaluación contra un resultado ya publicado.

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
Domótica en Español: Evaluación Automática de Doce Modelos Sub-2B}
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

- [ ] `00_abstract.tex` — `\begin{abstract}...\end{abstract}` + `\keywords{...}`, **solo en español** (A8: no va abstract en inglés). Debe declarar, con estas cuatro piezas: (a) que se evalúan **12** SLM abiertos de menos de 2B en interpretación de comandos de domótica en español rioplatense; (b) que la verificación es **totalmente automática** en dos etapas, coincidencia textual más un juez LLM con categorías cerradas, sin etiquetado manual; (c) que se reportan **dos** métricas, exactitud estricta y laxa, y que la brecha entre ambas cuantifica cuánto penaliza la coincidencia exacta a respuestas semánticamente correctas; (d) que todo corre en CPU con 2 núcleos y 8 GB, un contenedor por modelo. Sin números concretos: los resultados van en §4.
- [ ] `01_introduccion.tex` — mantener la motivación del original (SLM locales para domótica, privacidad, hardware modesto) y agregar el aporte nuevo: el original clasificó a mano sus 73 respuestas incorrectas, lo que no escala a un roster de este tamaño ni es reproducible; este trabajo automatiza ese juicio. Cerrar con las contribuciones enumeradas: roster de 12 modelos 2026 evaluados (de un registro de 15, con 3 excluidos: 2 por acceso restringido no otorgado y 1 por compatibilidad de versión no verificada), pipeline de verificación automática en dos etapas, doble métrica estricta/laxa, protocolo reproducible con aislamiento por contenedor, y continuidad con el estudio anterior a través de tres modelos comunes (antes dos) que cubren ambos tiers y dos familias.
- [ ] `02_trabajos_relacionados.tex` — reutilizar el texto del original (las 9 referencias siguen siendo válidas) y **agregar un párrafo nuevo sobre LLM-as-judge**, que es la técnica central que se incorpora: mencionar que la evaluación con modelos como jueces se usa cuando la coincidencia exacta subestima el desempeño, y que su limitación conocida es el sesgo de auto-favorecimiento — lo que enlaza con §6. Agregar a `refs.bib` **al menos una** referencia de LLM-as-judge y una de los modelos nuevos que se evalúan (familias LFM2.5 / Granite 4.0 / Qwen3.5 / OLMo-2). Las referencias nuevas **no deben deanonimizar**: nada de autocitas.

### Tarea 3 — Metodología (la sección que más cambia)

- [ ] `03_metodologia.tex` con `\section{Metodología}` y estas subsecciones:

**3.1 Modelos evaluados** — `\input{tablas/tabla1_modelos}` (12 filas, generada desde `roster_activo()`). Explicar el criterio de selección: modelos abiertos de dos tiers (sub-1B y 1–2B, 6 por tier); declarar como **limitación explícita** que el registro `MODELOS_2026` considera 15 modelos y que **3 quedaron fuera del roster evaluado**, por **dos causas distintas** que no deben mezclarse: 2 (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) porque el acceso de descarga no fue otorgado —repositorios marcados como gated con aprobación manual pendiente—, y 1 (`Qwen3.5-2B`) porque su compatibilidad de versión con `transformers` **no fue verificada** (no hay evidencia de que falle bajo la versión del grupo B; solo falta la confirmación positiva); ninguna de las tres exclusiones es una decisión metodológica; remitir a 3.6 para el detalle de reproducibilidad. Declarar también que los **12 modelos evaluados son todos de repositorios públicos**, sin restricción de licencia ni token.

Requisito nuevo (**RF19**, continuidad con la Tabla 2 publicada): declarar que `Qwen2.5-1.5B-Instruct` se **reincorporó deliberadamente** al roster activo para sostener la línea de continuidad con el estudio anterior. Los modelos presentes en **ambos** estudios pasan de **2 a 3**: `SmolLM2-360M-Instruct` (0.36 B), `Qwen2.5-1.5B-Instruct` (1.54 B) y `SmolLM2-1.7B-Instruct` (1.71 B). `Qwen2.5-1.5B-Instruct` es la única ancla del tier 1–2B de una familia distinta de SmolLM2, así que las tres anclas cubren ahora **ambos tiers y dos familias**. Cuidado editorial: **no** decir que la continuidad "se había perdido" y ahora "se restaura" —`SmolLM2` en sus dos tamaños siempre estuvo en ambos estudios—; la afirmación correcta es que las anclas pasan de 2 a 3.

Nota de corrección de identificador, para que nadie la reintroduzca: el ID `Qwen/Qwen2.5-1.7B-Instruct` **no existe** (HTTP 404 al resolverlo); el modelo real es `Qwen/Qwen2.5-1.5B-Instruct` (HTTP 200). Verificar que ni el `.tex` ni `refs.bib` contengan jamás `Qwen2.5-1.7B`.

Señalar que `SmolLM2` aparece en sus dos tamaños como parte de esa continuidad. **No** afirmar que ningún modelo se prompteó por *raw completion*: los 12 evaluados resuelven a `chat_template` (ver 3.3).

**3.2 Dataset de comandos** — sin cambios de fondo respecto del original: los mismos 32 comandos, el mismo esquema de 5 campos. Aclarar explícitamente que el dataset **no se modificó** para este estudio y que las categorías lingüísticas de §4.3 ya **no** se asignan a mano (ver 3.5).

**3.3 Protocolo de prompting y esquema de salida** — el cambio de **RF1**. Transcribir la cláusula taxativa en un `verbatim` o `quote` y explicar la decisión: los campos con valores entre paréntesis son categóricos cerrados y esos valores son los únicos outputs aceptados **textualmente**; usar un sinónimo es una violación de formato. Explicar el corolario metodológico: esto endurece deliberadamente el protocolo respecto del estudio anterior, así que la exactitud estricta **no es comparable** con la del paper original. Documentar también **RF2**: el modo de prompting (`chat_template` o *raw completion*) se decide en tiempo de ejecución por capacidad del tokenizer, nunca por identificador; la detección de Phase 3 estableció que los **12 modelos del roster activo resuelven a `chat_template`**, incluidos los dos LFM2.5 base, que un diagnóstico inicial suponía sin plantilla de chat. El modelo reincorporado `Qwen2.5-1.5B-Instruct` también resuelve a `chat_template`, confirmado por su propia sonda (`PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template`), no por inferencia de familia. La ruta de *raw completion* se conserva en el código como fallback por capacidad (cubierta por un test sintético), pero **ningún modelo evaluado la ejerce**: el paper no puede afirmar que algún modelo se prompteó por *raw completion*, porque no ocurrió.

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

**3.6 Entorno de ejecución y reproducibilidad** — **RF4** y **RF5**. Hardware: CPU, 2 núcleos, 8 GB, sin GPU, igual que el estudio original para que las latencias sean comparables. Aislamiento: una imagen de contenedor por modelo, ejecutadas de a una, compartiendo la caché de pesos. Y el punto que el usuario pidió explícitamente: `\input{tablas/tabla5_versiones}` con la matriz de versiones **por grupo**, más un párrafo que explique que hay **dos grupos de versión de `transformers`, mutuamente excluyentes sobre el roster** (grupo A resuelve 4.57.6 con **9** modelos, grupo B resuelve 5.14.1 con **3** modelos): ninguna versión mayor única cubre los 12 modelos evaluados. Nombrar qué modelo forzó cada grupo (`granite-4.0-350m` necesita el grupo A porque falla bajo 5.14.1; `LFM2.5-230M`, `LFM2.5-350M` y `Qwen3.5-0.8B` necesitan el grupo B porque fallan bajo 4.57.6) y declarar que esto es un confusor conocido para la comparación de latencia que se retoma en §6. La evidencia más nítida a citar: el grupo B resuelve exactamente a la versión (5.14.1) que rompe al modelo del grupo A — la prueba de que ninguna versión única sirve para todo el roster.

Agregar en 3.6, con el mismo nivel de detalle, la justificación de **RNF6/F11** (ejecución secuencial con *pinning* de núcleos), porque es un requisito de reproducibilidad y de comparabilidad de latencia, no un detalle de implementación: toda invocación de `docker run` de las 12 corridas del barrido y de la del juez fija `--memory=8g --cpus=2 --cpuset-cpus=0-1`, con **el mismo par de núcleos en las doce corridas y en la del juez**. Explicar por qué los tres flags no son redundantes entre sí: `--cpus` es una **cuota** del planificador CFS, no una reserva (permite migración entre núcleos); `--memory` es un **techo**, no una reserva (Docker solo ofrece la reserva blanda `--memory-reservation`); `--cpuset-cpus` sí **fija** los núcleos y elimina tanto la competencia por tiempo de CPU como la variabilidad por migración. Pero la inferencia de LLM en CPU está limitada por **ancho de banda de memoria**, y ni la caché L3 ni el bus de memoria pueden particionarse por contenedor, así que correr 4 modelos en paralelo habría inflado la latencia por comando de forma **invisible** en la tabla final. Por eso se eligió ejecución **secuencial**, para que la latencia siga siendo comparable con el paper original, a costa de ~7 h de reloj en vez de ~2 h. Aclarar que la exactitud, la taxonomía y la equivalencia semántica **no** dependen de esta decisión (decodificación determinista a temperatura 0): solo la latencia lo hace.

- [ ] **No** repetir la narrativa superada de que una única versión mayor de `transformers` elimina el confusor de versiones divergentes: esa premisa (la del commit `d6c7581`) fue refutada por la evidencia de Phase 3 y no puede aparecer en el paper.
- [ ] Cerrar 3.6 con el párrafo de **RF18** (reproducibilidad y modelos excluidos, redactado para envío ciego, sin datos identificatorios y sin mencionar ningún token concreto), reescrito a **3 de los 15 modelos del registro** con **dos causas**: 2 (`gemma-3-270m-it` y `Llama-3.2-1B-Instruct`) quedaron fuera del barrido porque el acceso de descarga no fue otorgado —no por una decisión metodológica—; sus repositorios exigirían aceptar una licencia propia y disponer de un token de HuggingFace para descargarse. El tercero, `Qwen3.5-2B`, quedó fuera porque su **compatibilidad de versión no fue verificada** —falla bajo la versión del grupo A, pero no hay evidencia de que falle ni de que funcione bajo la versión del grupo B; nunca se obtuvo la confirmación positiva— y tampoco es una decisión metodológica. **Redacción prudente, sin excepción:** no escribir en ningún punto que `Qwen3.5-2B` falla bajo 5.14.1 o que es incompatible con esa versión; la formulación correcta es "compatibilidad no verificada". Todo esto es irrelevante para reproducir *este* estudio, porque **los 12 modelos evaluados son todos de repositorios públicos**: reproducir el barrido publicado no requiere token ni aceptar ninguna licencia.

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
import re
from pathlib import Path
t = Path('paper/02_reescrito/secciones/03_metodologia.tex').read_text(encoding='utf-8').lower()
faltan = [k for k in ('taxativ','sinónimo','chat_template','estricta','laxa',
                      'greedy','contenedor','transformers','juez','categor',
                      'acceso restringido','5.14.1','4.57.6',
                      'qwen2.5-1.5b','no verificada','cpuset','secuencial')
          if k not in t]
assert not faltan, faltan
# 'raw completion' puede aparecer describiendo el fallback por capacidad (no ejercido),
# pero NUNCA afirmando que algun modelo del roster evaluado se prompteo asi.
prohibido = re.search(r'(lfm2\.5-230m|lfm2\.5-350m|modelos base)[^.]{0,80}raw completion', t)
assert not prohibido, 'el paper afirma que un modelo del roster se prompteo por raw completion'
# Redaccion prudente: nunca decir que Qwen3.5-2B falla bajo 5.14.1 ni que es incompatible con ella.
prohibido_qwen = re.search(r'qwen3\.5-2b[^.]{0,120}(falla|fall[oó]|incompatib)[^.]{0,40}5\.14\.1', t)
assert not prohibido_qwen, 'el paper afirma que Qwen3.5-2B falla/es incompatible bajo 5.14.1: viola la redaccion prudente'
assert 'qwen2.5-1.7b' not in t, 'el ID inexistente Qwen2.5-1.7B aparece en la metodologia'
print('metodologia cubre: prompt taxativo, chat_template, doble metrica, juez, docker, versiones, exclusion, continuidad, pinning')
"

# 6. Abstract solo en español
python -c "
from pathlib import Path
t = Path('paper/02_reescrito/secciones/00_abstract.tex').read_text(encoding='utf-8')
assert t.count('begin{abstract}') == 1, 'hay mas de un abstract'
print('abstract unico (español) OK')
"

# 6b. El ID inexistente Qwen2.5-1.7B no aparece en ningun .tex ni en refs.bib
grep -riq 'qwen2\.5-1\.7b' paper/02_reescrito/ \
  && echo "FALLA: aparece el ID inexistente Qwen2.5-1.7B" || echo "sin Qwen2.5-1.7B OK"

# 7. Nada fuera de alcance
git diff --name-only main | grep -v '^paper/02_reescrito/' \
  && echo "OJO: cambios fuera del alcance" || echo "alcance respetado"
```

## Acceptance criteria

- **Dado** `paper/02_reescrito/`, **entonces** tiene el layout de F9: `main.tex`, `refs.bib`, `llncs.cls`, `splncs03.bst`, 9 archivos en `secciones/`, 5 en `tablas/` y las 2 figuras en `figuras/`.
- **Dado** `main.tex`, **entonces** usa el preámbulo congelado, hace `\input` de las nueve secciones en orden, y su bloque de autor e institución está **anonimizado**, con `\hypersetup` forzando `pdfauthor`, `pdfsubject` y `pdfkeywords` vacíos.
- **Dado** `latexmk -pdf -halt-on-error main.tex`, **entonces** termina con código 0 y produce `main.pdf`, aun con las cinco secciones que todavía son stubs.
- **Dado** `python scripts/check_anonimato.py paper/02_reescrito`, **entonces** sale con **código 0**, revisando tanto los `.tex` como el texto y los metadatos del `main.pdf` construido.
- **Dado** el abstract, **entonces** está solo en español, en un único bloque `abstract`, y menciona los **12** modelos evaluados, la verificación automática en dos etapas y las dos métricas, sin dar cifras de resultados.
- **Dado** `02_trabajos_relacionados.tex`, **entonces** incluye un párrafo sobre LLM-as-judge con al menos una referencia nueva en `refs.bib`, y ninguna referencia nueva deanonimiza a los autores.
- **Dado** `03_metodologia.tex`, **entonces** hace `\input` de `tablas/tabla1_modelos` y `tablas/tabla5_versiones` (ningún número escrito a mano, 12 filas en ambas) y cubre explícitamente: la cláusula taxativa transcripta y su consecuencia sobre la comparabilidad con el estudio anterior; que la detección por capacidad estableció `chat_template` para los 12 modelos evaluados, incluido `Qwen2.5-1.5B-Instruct` confirmado por su propia sonda (sin afirmar que ninguno se prompteó por raw completion); la definición precisa de exactitud estricta y laxa; el pipeline de dos etapas con la regla de selección del juez y la decodificación greedy; las siete etiquetas y las seis categorías cerradas con su política de reintento y respaldo; el entorno de 2 núcleos / 8 GB con un contenedor por modelo; y la limitación de acceso/compatibilidad que excluyó a 3 de los 15 modelos del registro.
- **Dado** §3.1, **entonces** declara que `Qwen2.5-1.5B-Instruct` se reincorporó **deliberadamente** al roster para sostener la continuidad con la Tabla 2 publicada, nombra las tres anclas (`SmolLM2-360M-Instruct`, `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`), y enmarca el paso de 2 a 3 anclas como cobertura de **ambos tiers y dos familias**; y aclara que el ID `Qwen/Qwen2.5-1.7B-Instruct` no existe (HTTP 404) y que el ID real es `Qwen/Qwen2.5-1.5B-Instruct`.
- **Dado** §3.6, **entonces** declara los dos grupos de versión de `transformers` (mutuamente excluyentes sobre el roster, **9** en el grupo A y **3** en el grupo B), qué modelo forzó cada uno y por qué, citando `tab:versiones`; declara, sin datos identificatorios ni mención de ningún token concreto, que **3 de los 15 modelos del registro quedaron fuera del barrido** por dos causas —2 por acceso restringido no otorgado, 1 (`Qwen3.5-2B`) por compatibilidad de versión no verificada, nunca "falla bajo 5.14.1"— y que los 12 evaluados no requieren token ni licencia para reproducirse; y desarrolla la justificación de **RNF6/F11** para la ejecución secuencial con `--cpuset-cpus` fijo e idéntico en las 12 corridas y en el juez (cuota vs. techo vs. *pinning*, límite de ancho de banda de memoria, ~7 h vs. ~2 h, y que solo la latencia se ve afectada).
- **Dado** `03_metodologia.tex`, **entonces** no afirma en ningún punto que `LFM2.5-230M` o `LFM2.5-350M` se prompteó por raw completion (la sección 04_resultados, del subtask 15, hereda la misma prohibición).
- **Dado** cualquier archivo bajo `paper/02_reescrito/`, **entonces** no contiene el ID inexistente `Qwen2.5-1.7B` ni la afirmación de que `Qwen3.5-2B` falla o es incompatible bajo `transformers` 5.14.1.
- **Dado** `git diff --name-only main`, **entonces** los únicos cambios están bajo `paper/02_reescrito/`.
