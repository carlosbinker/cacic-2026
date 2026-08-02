# Reejecución del experimento con roster 2026, verificación automática en dos etapas y reescritura del paper en LaTeX

**Goal:** Rehacer el estudio comparativo de SLMs para interpretación de comandos de domótica con un roster de 14 modelos nuevos, reemplazar la clasificación manual de errores por un pipeline automático de dos etapas (coincidencia textual + juez LLM con categorías cerradas), aislar cada modelo en su propia imagen Docker, y reescribir el paper como un conjunto de archivos `.tex` LNCS listos para envío ciego a CACIC 2026.

**Architecture:** Todo el código nuevo vive en módulos `*_2026.py` paralelos a los existentes; los módulos legacy (`models.py`, `prompt.py`, `metrics.py`, `run_evaluation.py`, `generate_figures.py`), los resultados publicados (`data/resultados_experimento_*`) y `tests/test_metricas.py` quedan **congelados y verdes**. El barrido corre un modelo por contenedor, secuencialmente, escribiendo un CSV por modelo bajo `data/2026/detalle/` para que sea reanudable. Consolidado el barrido, se elige el juez de forma determinista (mejor exactitud estricta, desempate por tamaño) y ese mismo modelo etiqueta (a) cada respuesta incorrecta con una taxonomía cerrada de 7 etiquetas y (b) cada uno de los 32 comandos con una de 6 categorías lingüísticas. De ahí salen dos métricas titulares —exactitud estricta y laxa— cuya brecha es en sí misma un resultado. El paper se genera con tablas `.tex` emitidas por script desde los JSON de resultados.

**Tech stack:** Python 3.11, `transformers` (versión fijada por modelo), PyTorch CPU (bfloat16, greedy), pandas, matplotlib, Docker (una imagen por modelo, `--memory=8g --cpus=2`), LaTeX LNCS (`llncs.cls` + `splncs03.bst`) compilado con MiKTeX/`latexmk`, pytest.

---

## 1. Original specification

```
Tareas:

- Usar un sub-agente para convertir extraer el contenido de paper_cacic_LNCS_word.docx en un .tex que replique el formato exacto (formato de CACIC 2026, investigar la plantilla .tex en la fuente primaria). Usar un .tex para cada una de las secciones.
- Analizar el contenido del repositorio.
- Plantear como reproducir el experimento y reescribir el paper con las siguientes consideraciones:
  - En lugar de los modelos propuestos, se usan:
        ID del modelo	Params reales	Descarga	Licencia
        Sub-1B			
        LiquidAI/LFM2.5-230M	230M	HF, abierta	LFM Open License v1.0
        LiquidAI/LFM2.5-350M	350M	HF, abierta	LFM Open License v1.0
        ibm-granite/granite-4.0-350m	350M	HF, abierta	Apache 2.0
        ibm-granite/granite-4.0-h-350m	340M	HF, abierta	Apache 2.0
        Qwen/Qwen3.5-0.8B	0.8B	HF, abierta	Apache 2.0
        HuggingFaceTB/SmolLM2-360M-Instruct †	360M	HF, abierta	Apache 2.0
        google/gemma-3-270m-it †	270M	HF, gated	Gemma Terms of Use
        1–2B			
        LiquidAI/LFM2.5-1.2B-Instruct	1.2B	HF, abierta	LFM Open License v1.0
        ibm-granite/granite-4.0-1b	1.6B	HF, abierta	Apache 2.0
        ibm-granite/granite-4.0-h-1b	1.5B	HF, abierta	Apache 2.0
        Qwen/Qwen3.5-2B	2B	HF, abierta	Apache 2.0
        allenai/OLMo-2-0425-1B-Instruct †	1B	HF, abierta	Apache 2.0
        HuggingFaceTB/SmolLM2-1.7B-Instruct †	1.7B	HF, abierta	Apache 2.0
        meta-llama/Llama-3.2-1B-Instruct †	1B	HF, gated	Llama 3.2 Community License

        además del modelo SmallModelLanguage que ya es usado en el paper original. (en sus 2 tamaños)s
  - La verificación de los resultados se hace de forma automática y no manual:
    - En un primer paso, se verifican las coincidencias textuales entre el output esperado y el obtenido
    - En un segundo paso, se evaluan semanticamente las diferencias entre lo esperado y lo obtenido. Para esto, dado que son inputs chicos, usa el modelo que mejor puntuación haya obtenido en la primera etapa (si hay empates, el más grande. Esto debe ser parte de un solo pipeline automatizado). No debe ser abierto sino que deben haber categorías definidas. Usar las categorías de la versión actual del paper.
  - Correr cada modelo en un docker independiente para maximizar el aislamiento. Correr de a uno a la vez.
  - Incluí en el prompt que se les da a los modelos la especificación explicita y taxativa de que los campos con valores entre parentesis son campos categoricos y esos valores entre parentesis son, textualmente, los unicos outputs aceptados. Trata el usar sinonimos como una violación del formato.

- Crea el set de archivos .tex (un main y uno para cada sección) con la versión re-escrita del paper, siguiendo la estructura del original pero con estos cambios propuestos.
- Cada set de archivos .tex debe ir en una carpeta dedicada, por orden.
- Crea una rama de git dedicada para este refactor y haz commits periodicos a modo de checkpoint.
- Haz las preguntas lo antes posible.
```

### Delta 2026-08-02

# Delta 01 — los modelos gated vuelven al roster

## Qué cambia

`02-interview.md` §A1 dice: *"DROP `google/gemma-3-270m-it` and `meta-llama/Llama-3.2-1B-Instruct`
— no `HF_TOKEN` is available"*. **Eso queda anulado.** El usuario creó un token de HuggingFace y
ambos modelos **vuelven al roster**.

Verificado por root, no asumido:

```
200  google/gemma-3-270m-it
200  meta-llama/Llama-3.2-1B-Instruct
200  whoami-v2
```

Es decir: el token es válido **y** las licencias (Gemma Terms of Use, Llama 3.2 Community License)
ya están aceptadas en la cuenta. No hay paso manual pendiente ni riesgo de 401 a mitad del barrido.

La instrucción que root le dio al agente de Phase 1 — *"no auth step should appear anywhere in the
plan"* — también queda anulada: ahora **sí** hace falta plomería de credenciales, acotada a lo de
abajo.

## Consecuencias concretas para el plan

1. **Roster**: +2 modelos. El barrido, la matriz de imágenes Docker, las figuras y las tablas se
   dimensionan con el roster ampliado. Todo lo demás de §A1 sigue vigente: los Qwen2.5 siguen
   afuera, y los LFM2.5 base siguen adentro con ruta de raw-completion.
2. **Gemma 3 necesita `transformers >= 4.50.0`** (único mínimo confirmado por la investigación de
   Phase 1). Es exactamente el caso que motiva §A6: entra en la matriz de versiones documentada,
   con su motivo.
3. **Plomería del token** — el token vive en `.env` en la raíz del repo, ya creado y verificado
   como ignorado por git:
   - `.gitignore` ahora cubre `.env`, `*.env`, `.hf_token`.
   - El barrido lo pasa a los contenedores por entorno: `docker run --env-file .env ...` o
     `-e HF_TOKEN`. **Nunca** como `ARG`/`ENV` en un `Dockerfile` ni como capa de imagen.
   - Solo lo reciben las dos imágenes que lo necesitan; los demás modelos corren sin credenciales.
   - Si falta `HF_TOKEN`, el barrido debe fallar temprano con un mensaje claro, no a mitad de la
     descarga.
4. **Prohibición dura**: el token **no puede aparecer** en ningún archivo versionado — ni en los
   `TODO_<NN>_*.md`, ni en el README, ni en la matriz de versiones, ni en logs commiteados, ni en
   `data/2026/`. Los archivos del plan lo referencian **solo** como `$HF_TOKEN`. Esto es criterio
   de aceptación verificable (`git grep -i 'hf_[A-Za-z]'` debe dar vacío).
5. **Reproducibilidad / paper**: dos de los modelos requieren aceptar una licencia y un token para
   descargarse. Eso limita la reproducibilidad de terceros y **debe decirse** en la sección de
   reproducibilidad y en §6 Amenazas a la Validez, junto con la divergencia de versiones de §A6.
   Redactarlo sin datos identificatorios (submission ciega, §A8).

## Nota de seguridad para el usuario (no es tarea del plan)

El token se pegó en texto plano en el chat. Funciona y no bloquea nada, pero conviene **rotarlo en
huggingface.co/settings/tokens cuando termine el barrido**, y darle scope de solo lectura si el
actual es de escritura. Sustituir el valor en `.env` alcanza; ningún archivo versionado lo contiene.

## 2. Formalized specification

### 2.1 Roster de modelos (14)

`Qwen2.5-0.5B-Instruct` y `Qwen2.5-1.5B-Instruct` se **eliminan** (superados por el par Qwen3.5). `google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct` **se mantienen**: son *gated*, pero hay un `$HF_TOKEN` válido y ambas licencias ya están aceptadas en la cuenta, así que se descargan sin intervención manual (ver §4, F11 para la plomería del token). `SmolLM2` aporta los "2 tamaños" del modelo del paper original (360M y 1.7B). Roster final:

| # | `nombre` | `hf_repo_id` | `params_b` | `tier` | prompting | `gated` |
|---|----------|--------------|-----------|--------|-----------|---------|
| 1 | `LFM2.5-230M` | `LiquidAI/LFM2.5-230M` | 0.23 | sub-1B | raw_completion (base) | no |
| 2 | `LFM2.5-350M` | `LiquidAI/LFM2.5-350M` | 0.35 | sub-1B | raw_completion (base) | no |
| 3 | `granite-4.0-350m` | `ibm-granite/granite-4.0-350m` | 0.35 | sub-1B | chat_template | no |
| 4 | `granite-4.0-h-350m` | `ibm-granite/granite-4.0-h-350m` | 0.34 | sub-1B | chat_template | no |
| 5 | `Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | 0.8 | sub-1B | chat_template | no |
| 6 | `SmolLM2-360M-Instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36 | sub-1B | chat_template | no |
| 7 | `gemma-3-270m-it` | `google/gemma-3-270m-it` | 0.27 | sub-1B | chat_template | sí |
| 8 | `LFM2.5-1.2B-Instruct` | `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2 | 1-2B | chat_template | no |
| 9 | `granite-4.0-1b` | `ibm-granite/granite-4.0-1b` | 1.6 | 1-2B | chat_template | no |
| 10 | `granite-4.0-h-1b` | `ibm-granite/granite-4.0-h-1b` | 1.5 | 1-2B | chat_template | no |
| 11 | `Qwen3.5-2B` | `Qwen/Qwen3.5-2B` | 2.0 | 1-2B | chat_template | no |
| 12 | `OLMo-2-0425-1B-Instruct` | `allenai/OLMo-2-0425-1B-Instruct` | 1.0 | 1-2B | chat_template | no |
| 13 | `SmolLM2-1.7B-Instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71 | 1-2B | chat_template | no |
| 14 | `Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` | 1.0 | 1-2B | chat_template | sí |

El `modo_prompting` de la última columna es la **expectativa**; el código lo decide en runtime por capacidad (`tokenizer.chat_template is None`), nunca por ID hardcodeado. Si la detección discrepa de la tabla, gana la detección y se corrige la tabla del paper.

### 2.2 Requisitos funcionales

- **RF1** — Prompt de sistema 2026: el prompt del paper más una cláusula **taxativa** que declare que los valores entre paréntesis son los únicos outputs aceptados textualmente y que usar sinónimos es una violación de formato (texto exacto congelado en §4, F2).
- **RF2** — Ruta de prompting por *raw completion* para tokenizers sin `chat_template`, detectada por capacidad.
- **RF3** — Barrido **reanudable por modelo**: un CSV por modelo; reejecutar salta los modelos ya completos salvo `--force`.
- **RF4** — **Una imagen Docker por modelo** (14), con su `transformers` fijado, compartiendo un único volumen de caché HF. Ejecución secuencial, `--memory=8g --cpus=2`, CPU-only. Las dos imágenes de modelos *gated* reciben `$HF_TOKEN` por entorno en runtime (§4, F11); las otras 12 corren sin credenciales.
- **RF5** — Toda divergencia de versión respecto del baseline debe quedar documentada (qué modelo la forzó, qué error evita) en `docker/README.md`, en el sitio del pin, y en el texto de metodología y de amenazas del paper. Lo mismo para cualquier uso de `trust_remote_code`.
- **RF6** — Etapa 1 (automática): coincidencia textual exacta campo a campo, como hoy.
- **RF7** — Selección determinista del juez: mayor `exact_match_pct` de la etapa 1; empate → mayor `params_b`; empate persistente → orden del roster. Decodificación greedy/temperatura 0 en etapa 1 y etapa 2.
- **RF8** — Etapa 2, trabajo A: el juez clasifica cada respuesta con `match_exact == False` en un subconjunto **no vacío** de las 7 etiquetas cerradas (§4, F4). Salida cerrada, no abierta.
- **RF9** — Etapa 2, trabajo B: el juez clasifica cada uno de los 32 comandos en **exactamente una** de las 6 categorías lingüísticas cerradas (§4, F4). Reemplaza el etiquetado manual de la Tabla 3; **no** se agrega columna al dataset.
- **RF10** — Dos métricas titulares: **exactitud estricta** (`match_exact`) y **exactitud laxa** (`match_exact` o etiqueta en `{sin_error_semantico, uso_de_sinonimos}`). Ambas en la Tabla 2; la brecha se discute como resultado.
- **RF11** — Tabla 4 con las **5 categorías publicadas** (se separa `alucinacion_valor_unidad` de `valor_numerico_incorrecto`; el legacy las fusionaba) más las 2 etiquetas nuevas.
- **RF12** — Figuras rediseñadas: **barras horizontales agrupadas por tier**, nombres largos legibles, sin apiñamiento a 14 modelos.
- **RF13** — Tablas 1–5 del paper generadas por script desde los JSON de resultados hacia fragmentos `.tex` que `main.tex` hace `\input`.
- **RF14** — Dos árboles LaTeX: `paper/01_original/` (transcripción fiel del `.docx`) y `paper/02_reescrito/`, cada uno con `main.tex`, un `.tex` por sección y `refs.bib`, autocontenidos (`llncs.cls` y `splncs03.bst` copiados), y ambos compilando a PDF.
- **RF15** — **Envío ciego**: cero datos identificatorios en `paper/02_reescrito/` ni en su PDF (sin autores, afiliaciones, agradecimientos, financiamiento, URL del repositorio, ORCID, ni metadatos identificatorios). Verificado por script, no por inspección.
- **RF16** — El paper reescrito debe incorporar cuatro amenazas a la validez **nuevas**: sesgo de auto-favorecimiento del juez, incomparabilidad de los dos modelos base prompteados por raw completion, versiones divergentes de librería como confusor de la latencia, y reproducibilidad limitada por los dos modelos *gated*, que exigen aceptar una licencia y disponer de un token para descargarse.
- **RF17** — **Credenciales**: el token de HuggingFace vive solo en `.env` (ignorado por git) y se pasa a los dos contenedores *gated* con `docker run --env-file .env`. Nunca como `ARG`/`ENV` de Dockerfile, `--build-arg`, `COPY`, capa de imagen ni `login` interactivo. Su **valor** no puede aparecer en ningún archivo versionado. Si falta, el barrido falla **temprano** con un mensaje claro en español que no imprime el token.
- **RF18** — **Reproducibilidad declarada**: el paper reescrito debe decir, en la sección de metodología/reproducibilidad y en §6 Amenazas a la Validez, que 2 de los 14 modelos son *gated* y que reproducir el barrido completo exige aceptar sus licencias y usar un token propio de HuggingFace. Redactado sin datos identificatorios (envío ciego, RF15).

### 2.3 Requisitos no funcionales

- **RNF1** — Hardware: la misma máquina Windows con Docker Desktop, CPU-only, `--memory=8g --cpus=2`, para preservar la comparabilidad de la tabla de latencia con el paper original. Sin GPU.
- **RNF2** — Presupuesto: barrido secuencial de ~4–8 h y ~20–40 GB de descargas. Debe tolerar interrupciones (RF3).
- **RNF3** — Determinismo: `do_sample=False`, `temperature=None`, `top_p=None` en las dos etapas.
- **RNF4** — Los tests existentes (`pytest -q`) quedan verdes en todo momento, en cada commit.
- **RNF5** — Un commit por subtask, en la rama `feat/reescritura-experimento-2026` (ya creada desde `main`).

### 2.4 Fuera de alcance (out of scope)

- Cualquier flujo de login interactivo (`huggingface-cli login`, `huggingface_hub.login()`) o gestión de credenciales más allá de leer `$HF_TOKEN` del entorno. Rotar, emitir o almacenar tokens tampoco es parte del plan.
- Los modelos `Qwen2.5-*` del paper original.
- Modificar `data/resultados_experimento_detalle.csv`, `data/resultados_experimento_resumen.json`, `tests/test_metricas.py`, `data/dataset_comandos_domotica.csv` o los módulos legacy de `src/`.
- Abstract en inglés (el paper es solo en español).
- Chequeo automático del límite de 10 páginas.
- Commitear `paper_cacic_LNCS_word.docx` (queda *untracked*, en `.gitignore`).
- Ampliar, reequilibrar o reetiquetar a mano el dataset de 32 comandos.
- Uso de GPU, cuantización, o cualquier juez externo vía API.
- `/split-todo`: es un único DAG en una única rama.

## 3. Subtask DAG

| id | title | depends_on | file |
|----|-------|------------|------|
| 01 | Registro de modelos 2026 y vocabularios cerrados | [] | `TODO_01_registro-modelos-2026.md` |
| 02 | Prompt taxativo 2026 y ruta de raw completion | [01] | `TODO_02_prompt-y-raw-completion.md` |
| 03 | Harness de barrido reanudable por modelo | [01, 02] | `TODO_03_harness-barrido-reanudable.md` |
| 04 | Imágenes Docker por modelo y matriz de versiones | [01, 03] | `TODO_04_docker-por-modelo.md` |
| 05 | Ejecución del barrido completo (14 modelos) | [04] | `TODO_05_ejecucion-barrido.md` |
| 06 | Consolidación de etapa 1 y selección del juez | [05] | `TODO_06_consolidacion-y-juez.md` |
| 07 | Juez LLM: taxonomía de errores y categorías lingüísticas | [06] | `TODO_07_juez-llm.md` |
| 08 | Ejecución de la etapa 2 | [07] | `TODO_08_ejecucion-juez.md` |
| 09 | Métricas 2026: estricta, laxa, Tablas 2/3/4 | [08] | `TODO_09_metricas-2026.md` |
| 10 | Figuras horizontales agrupadas por tier | [09] | `TODO_10_figuras-2026.md` |
| 11 | Generador de fragmentos `.tex` de tablas | [09] | `TODO_11_generador-tablas-tex.md` |
| 12 | Verificador de anonimato para envío ciego | [] | `TODO_12_verificador-anonimato.md` |
| 13 | Transcripción fiel del `.docx` a `paper/01_original/` | [] | `TODO_13_transcripcion-original.md` |
| 14 | Paper reescrito: andamiaje y secciones 1–3 | [10, 11, 12, 13] | `TODO_14_paper-secciones-1-3.md` |
| 15 | Paper reescrito: secciones 4–7, build y anonimato | [14] | `TODO_15_paper-secciones-4-7.md` |

**Concurrencia.** `12` y `13` no tienen dependencias y sus conjuntos de archivos son disjuntos de todo lo demás y entre sí: pueden ejecutarse en cualquier momento, en paralelo con `01`–`11`. `10` y `11` dependen ambos de `09` pero tocan archivos disjuntos (`src/generate_figures_2026.py` + `figures/2026/` vs. `src/generate_tex_tables.py` + `paper/02_reescrito/tablas/`): pueden correr concurrentemente. El resto es una cadena estricta. `05` y `08` son las dos tareas de ejecución largas y bloquean todo lo que va después.

## 4. Frozen interfaces / contracts

Estos contratos se diseñan una vez, acá. Si un implementador descubre que uno está mal o es insuficiente, **para**, se vuelve a congelar en esta sección, se actualizan los subtasks ya escritos que dependían de él, y recién entonces continúa. Nunca se renegocian ad hoc.

### F0 — Invariantes congelados del repositorio

Estos archivos son **inmutables** en toda la rama. Cualquier subtask que los modifique está mal por construcción:

- `data/resultados_experimento_detalle.csv`
- `data/resultados_experimento_resumen.json`
- `data/dataset_comandos_domotica.csv`
- `tests/test_metricas.py`
- `src/models.py`, `src/prompt.py`, `src/scoring.py`, `src/schema.py`, `src/metrics.py`, `src/run_evaluation.py`, `src/generate_figures.py`, `src/validar_contra_resultados_originales.py`
- `figures/fig1_exactitud_latencia.png`, `figures/fig2_exactitud_por_campo.png`

Corolario: **todo el código nuevo va en módulos nuevos**, no en ediciones de los existentes. `src/scoring.py` y `src/schema.py` se **importan y reutilizan** tal cual (`comparar_campos`, `extraer_json`, `fila_a_ground_truth`, `CAMPOS_ESQUEMA`); `src/prompt.py::construir_prompt_usuario` y `SYSTEM_PROMPT_PAPER` también se reutilizan tal cual.

### F1 — Convenciones compartidas

- **Layout de imports.** Los módulos de `src/` se importan planos (`from models_2026 import MODELOS_2026`), igual que hoy, porque se ejecutan como `python src/<script>.py`.
- **Tests.** Van en `tests/test_<modulo>.py` y arrancan exactamente con el preámbulo que ya usa `tests/test_metricas.py`:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
  ```
  Se corren con `pytest -q` desde la raíz del repo. No se agrega `conftest.py` ni `pytest.ini`.
- **Raíz del repo.** Todo módulo de `src/` la calcula igual: `RAIZ = Path(__file__).resolve().parent.parent`.
- **Errores.** Precondición violada o etiqueta fuera de vocabulario → `ValueError` con mensaje en español que incluya el valor ofensor. Nada de `assert` para validación en runtime, nada de fallar en silencio.
- **Escritura de artefactos.** Siempre `Path(...).parent.mkdir(parents=True, exist_ok=True)` antes de escribir; JSON con `indent=2, ensure_ascii=False`; CSV con `index=False`.
- **Idioma.** Docstrings, comentarios, mensajes de error y prosa del paper en español. Identificadores de código y claves de datos en español sin tildes ni ñ (igual que `schema.py`).
- **Logging.** `print()` a stdout, igual que los scripts existentes; sin librería de logging.
- **Dueño único de `.gitignore`.** El subtask **01** es el único autorizado a editar `.gitignore`. El archivo **ya está commiteado** (en `8acfefd`) con las entradas de caché HF, credenciales (`.env`, `*.env`, `.hf_token`), el `.docx` y la plantilla `LaTeX2e (1)/`. El subtask 01 por lo tanto **verifica y completa, nunca reescribe ni reordena**: comprueba que esas entradas estén y agrega **únicamente lo que falte** (hoy: los auxiliares de LaTeX `*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`, `*.synctex.gz`, `*.fls`, `*.fdb_latexmk`, `*.toc`). Ningún otro subtask lo toca: los subtasks 05 y 13 solo **verifican** que las entradas estén, y paran si faltan.
- **`git add` acotado.** Cada subtask agrega solo las rutas que declara en su frontmatter, nunca `git add -A` ni `git add .`, para que ningún artefacto pesado o no versionable entre por accidente.

### F2 — Prompting: `src/prompt_2026.py`

```python
from typing import Literal

ModoPrompting = Literal["chat_template", "raw_completion"]

CLAUSULA_TAXATIVA: str
SYSTEM_PROMPT_2026: str
RAW_TEMPLATE: str

def detectar_modo(tokenizer) -> ModoPrompting: ...
def construir_entrada(tokenizer, comando: str) -> tuple[dict, ModoPrompting]: ...
```

`SYSTEM_PROMPT_2026` es, literalmente, `SYSTEM_PROMPT_PAPER + "\n\n" + CLAUSULA_TAXATIVA + "\n\n" + _SEGUNDO_EJEMPLO`, donde `SYSTEM_PROMPT_PAPER` se **importa** de `src/prompt.py` (no se copia). El texto congelado de `CLAUSULA_TAXATIVA` es exactamente:

```
IMPORTANTE: los campos "intent", "dispositivo", "ubicacion" y "unidad" son campos CATEGORICOS CERRADOS. Los valores listados entre parentesis mas arriba son, textualmente, los unicos outputs aceptados para esos campos: copialos exactamente como aparecen, sin tildes, sin mayusculas, sin plural y sin reformular. Usar un sinonimo, una traduccion o cualquier variante que no este literalmente en esa lista (por ejemplo "televisor" o "tele" en lugar de "tv") es una violacion del formato y la respuesta se considera incorrecta. Si ningun valor de la lista aplica, usa null.
```

y el de `_SEGUNDO_EJEMPLO`:

```
Ejemplo 2: Comando: "Bajale un poco a la luz del living."
{"intent": "ajustar", "dispositivo": "luz", "ubicacion": "living", "valor": null, "unidad": null}
```

`RAW_TEMPLATE` congelado (para modelos base, sin `chat_template`):

```
{system}

Comando: "{comando}"
JSON:
```

`detectar_modo` devuelve `"raw_completion"` si y solo si `getattr(tokenizer, "chat_template", None)` es `None` o cadena vacía. `construir_entrada` devuelve `(entrada, modo)` donde `entrada` es siempre un dict con al menos la clave `input_ids` (tensor `pt`), apto para `modelo.generate(**entrada)`:
- modo `chat_template` → `tokenizer.apply_chat_template([{"role":"system","content":SYSTEM_PROMPT_2026},{"role":"user","content":construir_prompt_usuario(comando)}], add_generation_prompt=True, return_tensors="pt", return_dict=True)`
- modo `raw_completion` → `tokenizer(RAW_TEMPLATE.format(system=SYSTEM_PROMPT_2026, comando=comando), return_tensors="pt")`

### F3 — Registro de modelos: `src/models_2026.py`

```python
from dataclasses import dataclass
from typing import Literal

Tier = Literal["sub-1B", "1-2B"]

BASELINE_TRANSFORMERS: str = "transformers>=4.57.0"

@dataclass(frozen=True)
class ModeloEvaluado2026:
    nombre: str                # clave única; valor de la columna `modelo` en todos los CSV
    hf_repo_id: str
    params_b: float
    tier: Tier
    transformers_pin: str      # spec pip exacto usado en la imagen de ese modelo
    trust_remote_code: bool
    gated: bool                # requiere licencia aceptada + $HF_TOKEN para descargarse
    motivo_pin: str            # "" si transformers_pin == BASELINE_TRANSFORMERS; si no, el porqué

MODELOS_2026: list[ModeloEvaluado2026]   # los 14 de §2.1, en ese orden

def slug(nombre: str) -> str: ...        # minúsculas; [^a-z0-9]+ -> "-"; sin guiones al borde
def por_nombre(nombre: str) -> ModeloEvaluado2026: ...   # ValueError si no existe
def por_tier(tier: Tier) -> list[ModeloEvaluado2026]: ...
def gated() -> list[ModeloEvaluado2026]: ...   # los que requieren $HF_TOKEN, en orden de roster
```

`src/models_2026.py` declara el **flag** `gated`; **no** lee el token ni menciona su valor.

**Invariantes congelados, verificados por test:** (a) `motivo_pin != ""` si y solo si `transformers_pin != BASELINE_TRANSFORMERS`; (b) exactamente **dos** modelos tienen `gated = True`, y son `google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct`. Valor inicial de los 14: `transformers_pin = BASELINE_TRANSFORMERS`, `trust_remote_code = False`, `motivo_pin = ""`. El subtask 04 es el único autorizado a cambiar los pines, y solo con evidencia empírica de fallo.

### F4 — Vocabularios cerrados: `src/taxonomia_2026.py`

```python
ETIQUETAS_ERROR: list[str] = [
    "confusion_intencion",
    "confusion_dispositivo",
    "confusion_ubicacion",
    "alucinacion_valor_unidad",
    "valor_numerico_incorrecto",
    "sin_error_semantico",
    "uso_de_sinonimos",
]
ETIQUETAS_EQUIVALENTES: frozenset[str] = frozenset({"sin_error_semantico", "uso_de_sinonimos"})

CATEGORIAS_LINGUISTICAS: list[str] = [
    "encendido_apagado_simple",
    "ajuste_con_valor_numerico",
    "ajuste_relativo_cualitativo",
    "multiples_dispositivos",
    "consulta_de_estado",
    "negacion_contexto",
]

ETIQUETAS_ERROR_DISPLAY: dict[str, str] = {
    "confusion_intencion": "Confusión de intención",
    "confusion_dispositivo": "Confusión de dispositivo",
    "confusion_ubicacion": "Confusión de ubicación",
    "alucinacion_valor_unidad": "Alucinación de valor/unidad",
    "valor_numerico_incorrecto": "Valor numérico incorrecto",
    "sin_error_semantico": "Sin error semántico",
    "uso_de_sinonimos": "Uso de sinónimos",
}
CATEGORIAS_DISPLAY: dict[str, str] = {
    "encendido_apagado_simple": "Encendido / apagado simple",
    "ajuste_con_valor_numerico": "Ajuste con valor numérico",
    "ajuste_relativo_cualitativo": "Ajuste relativo / cualitativo",
    "multiples_dispositivos": "Múltiples dispositivos",
    "consulta_de_estado": "Consulta de estado",
    "negacion_contexto": "Negación / contexto conversacional",
}

SEP_ETIQUETAS: str = ";"

def parsear_etiquetas(texto: str) -> list[str]: ...   # ValueError si vacío o fuera de vocabulario
def parsear_categoria(texto: str) -> str: ...         # ValueError si no es exactamente una válida
```

Las 5 primeras etiquetas son la Tabla 4 publicada (el legacy fusionaba la 4 y la 5; acá van separadas). Las 2 últimas existen solo para la etapa 2.

### F5 — Espacio de datos `data/2026/`

```
data/2026/
  detalle/<slug>.csv          # una por modelo (subtask 05)
  detalle_2026.csv            # consolidado (subtask 06)
  resumen_etapa1.json         # resumen sin métrica laxa (subtask 06)
  juez_seleccionado.json      # (subtask 06)
  etiquetas_errores.csv       # (subtask 08)
  categorias_comandos.csv     # (subtask 08)
  resumen_2026.json           # resumen final con estricta + laxa (subtask 09)
  taxonomia_2026.csv          # Tabla 4 (subtask 09)
  exactitud_por_categoria.csv # Tabla 3 (subtask 09)
```

**Columnas de `detalle/<slug>.csv` y de `detalle_2026.csv`** (orden exacto; las 15 primeras son idénticas al CSV legacy para poder reutilizar `scoring.py` sin cambios):

```
modelo,idx,comando,gt,pred_raw,pred_json,json_valido,parse_note,latencia_s,
match_intent,match_dispositivo,match_ubicacion,match_valor,match_unidad,match_exact,
modo_prompting,transformers_version
```

**`juez_seleccionado.json`:**
```json
{
  "modelo": "<nombre del roster>",
  "hf_repo_id": "<repo>",
  "params_b": 1.71,
  "exact_match_pct": 62.5,
  "criterio": "mayor exact_match_pct en etapa 1; desempate por mayor params_b; luego orden del roster",
  "empatados": ["<nombre>", "..."]
}
```

**`etiquetas_errores.csv`:** `modelo,idx,etiquetas,juez_raw,juez_parse_ok`
donde `etiquetas` es un subconjunto no vacío de `ETIQUETAS_ERROR` unido por `SEP_ETIQUETAS`. Solo contiene filas con `match_exact == False`.

**`categorias_comandos.csv`:** `idx,comando,categoria,juez_raw,juez_parse_ok`
exactamente 32 filas, `categoria ∈ CATEGORIAS_LINGUISTICAS`.

**Fallback de parseo (congelado).** Si la salida del juez no parsea contra el vocabulario cerrado, se reintenta **una** vez con el mismo prompt y `max_new_tokens` duplicado. Si vuelve a fallar: `juez_parse_ok = False` y se escribe la etiqueta de respaldo determinista — la primera etiqueta `confusion_*` cuyo `match_*` correspondiente sea falso, en el orden `intent, dispositivo, ubicacion`; si los tres coinciden, `"valor_numerico_incorrecto"`. Para categorías, el respaldo es `"encendido_apagado_simple"`. La cantidad de filas con `juez_parse_ok == False` se reporta en el paper.

**`resumen_2026.json`** — lista de dicts, un elemento por modelo, en orden de roster, con exactamente estas claves:
```
modelo, hf_repo_id, params_b, tier, modo_prompting, transformers_pin, n,
json_valido_pct, exact_match_pct, exact_match_laxo_pct, avg_latencia_s,
acc_intent_pct, acc_dispositivo_pct, acc_ubicacion_pct, acc_valor_pct, acc_unidad_pct
```
Porcentajes redondeados a 1 decimal, latencia a 3, igual que el legacy.

### F6 — Métricas: `src/metrics_2026.py`

```python
import pandas as pd

def calcular_resumen_2026(df_detalle: pd.DataFrame, df_etiquetas: pd.DataFrame) -> list[dict]: ...
def calcular_taxonomia_errores_2026(df_detalle: pd.DataFrame, df_etiquetas: pd.DataFrame) -> pd.DataFrame: ...
def calcular_exactitud_por_categoria(df_detalle: pd.DataFrame, df_categorias: pd.DataFrame) -> pd.DataFrame: ...
```

Definiciones congeladas:
- **estricta** = `match_exact`.
- **laxa** = `match_exact` **o** (las etiquetas de esa fila intersecan `ETIQUETAS_EQUIVALENTES`). La laxa es siempre ≥ la estricta; el test lo verifica.
- `calcular_taxonomia_errores_2026` devuelve columnas `modelo, total_incorrectas` + una columna `int` por cada una de las 7 `ETIQUETAS_ERROR` (conteos no excluyentes).
- `calcular_exactitud_por_categoria` devuelve columnas `categoria, n` + una columna por modelo (orden de roster) con el porcentaje de exactitud **estricta** dentro de esa categoría.

### F7 — Figuras: `src/generate_figures_2026.py`

Módulo **nuevo**; `src/generate_figures.py` queda intacto (F0) para que las figuras publicadas sigan siendo reproducibles. Conserva la convención de CLI de aquel: `--resumen`, `--sufijo`, más `--salida-dir` (default `figures/2026`). Barras **horizontales**, agrupadas y separadas visualmente por `tier` (`sub-1B` arriba, `1-2B` abajo), etiquetas de modelo completas y legibles. Salidas: `figures/2026/fig1_exactitud_latencia_2026<sufijo>.png` (panel izq. exactitud estricta y laxa superpuestas; panel der. latencia promedio) y `figures/2026/fig2_exactitud_por_campo_2026<sufijo>.png`.

### F8 — Tablas LaTeX: `src/generate_tex_tables.py`

Escribe en `paper/02_reescrito/tablas/` cinco fragmentos, cada uno un bloque `\begin{table}...\end{table}` autónomo, sin preámbulo, apto para `\input`:

| archivo | contenido | fuente |
|---|---|---|
| `tabla1_modelos.tex` | roster: modelo, params, tier, familia, modo de prompting | `models_2026.MODELOS_2026` |
| `tabla2_resultados_globales.tex` | JSON válido, exactitud estricta, exactitud laxa, latencia | `resumen_2026.json` |
| `tabla3_por_categoria.tex` | exactitud por categoría lingüística × modelo | `exactitud_por_categoria.csv` |
| `tabla4_taxonomia.tex` | taxonomía de 7 etiquetas × modelo | `taxonomia_2026.csv` |
| `tabla5_versiones.tex` | matriz de versiones: modelo, `transformers_pin`, `trust_remote_code`, motivo | `models_2026.MODELOS_2026` |

Etiquetas LaTeX congeladas: `\label{tab:modelos}`, `\label{tab:globales}`, `\label{tab:categorias}`, `\label{tab:taxonomia}`, `\label{tab:versiones}`. Todo texto que provenga de datos pasa por un escapador de LaTeX (`_escapar`) para `& % $ # _ { } ~ ^ \`.

### F9 — Layout de los papers

```
paper/01_original/                paper/02_reescrito/
  llncs.cls                         llncs.cls
  splncs03.bst                      splncs03.bst
  main.tex                          main.tex
  refs.bib                          refs.bib
  secciones/                        secciones/
    00_abstract.tex                   00_abstract.tex
    01_introduccion.tex               01_introduccion.tex
    02_trabajos_relacionados.tex      02_trabajos_relacionados.tex
    03_metodologia.tex                03_metodologia.tex
    04_resultados.tex                 04_resultados.tex
    05_discusion.tex                  05_discusion.tex
    06_amenazas.tex                   06_amenazas.tex
    07_conclusiones.tex               07_conclusiones.tex
    08_declaracion_ia.tex             08_declaracion_ia.tex
                                    tablas/     (generado por F8)
                                    figuras/    (copiadas de figures/2026/)
```

`llncs.cls` y `splncs03.bst` se **copian** desde `LaTeX2e (1)/` a cada carpeta para que el build sea autocontenido. `LaTeX2e (1)/` y `LaTeX2e (1).zip` quedan *untracked* (`.gitignore`).

Preámbulo congelado de ambos `main.tex`:
```latex
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-tabla,es-noquoting]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
```
Bibliografía: `\bibliographystyle{splncs03}` + `\bibliography{refs}`. Cada sección se incorpora con `\input{secciones/NN_nombre}`.

Comando de build congelado (desde la carpeta del paper):
```
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### F10 — Anonimato: `scripts/check_anonimato.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Hallazgo:
    archivo: str
    linea: int      # 0 si el hallazgo es de metadatos del PDF
    patron: str
    motivo: str
    fragmento: str

PATRONES_PROHIBIDOS: list[tuple[str, str]]   # (regex, motivo legible)

def revisar_tex(rutas: list[Path]) -> list[Hallazgo]: ...
def revisar_pdf(ruta_pdf: Path) -> list[Hallazgo]: ...
def revisar_carpeta(carpeta: Path) -> list[Hallazgo]: ...
```

CLI: `python scripts/check_anonimato.py paper/02_reescrito` → imprime los hallazgos y **sale con código 1** si hay alguno, 0 si no hay ninguno. Es la forma ejecutable de RF15 y se invoca en el `Verify` del subtask 15.

### F11 — Docker

```
docker/
  Dockerfile.modelo        # parametrizado por ARG TRANSFORMERS_PIN
  requirements-base.txt    # todo menos transformers
  build_all.py             # construye una imagen por modelo del roster
  run_sweep.py             # corre las 14 imágenes, de a una, en orden de roster
  README.md                # matriz de versiones + motivos (RF5)
```

Nombre de imagen congelado: `slm-domotica-2026:<slug>`. Invocación congelada por modelo, según `gated`:

Modelos **no gated** (12) — sin credenciales, idéntico a hoy:
```
docker run --rm --memory=8g --cpus=2 \
  -v <repo>/data:/app/data \
  -v <repo>/.hf_cache:/app/.hf_cache \
  slm-domotica-2026:<slug> \
  python src/run_sweep_2026.py --modelo "<nombre>"
```

Modelos **gated** (2) — se agrega **exclusivamente** `--env-file <repo>/.env`:
```
docker run --rm --memory=8g --cpus=2 \
  --env-file <repo>/.env \
  -v <repo>/data:/app/data \
  -v <repo>/.hf_cache:/app/.hf_cache \
  slm-domotica-2026:<slug> \
  python src/run_sweep_2026.py --modelo "<nombre>"
```
Un único volumen de caché HF compartido por todas las imágenes; los árboles de dependencias quedan aislados. `docker/README.md` debe contener una tabla `modelo | transformers_pin | trust_remote_code | ¿necesario? | motivo` que sea la **misma información** que la Tabla 5 del paper.

**Credenciales (congelado).** El token de HuggingFace vive **solo** en `.env` en la raíz del repo, ignorado por git. Se inyecta **únicamente en tiempo de ejecución** vía `docker run --env-file .env`, y **solo** a las dos imágenes con `gated = True`; las otras 12 corren sin credenciales. Está **prohibido**: declarar `ARG HF_TOKEN` o `ENV HF_TOKEN` en cualquier `Dockerfile`, pasar el token como `--build-arg`, hacer `COPY .env`, escribirlo en una capa de imagen, o invocar `huggingface-cli login` / `huggingface_hub.login()`. La única forma de consumirlo es leer `os.environ["HF_TOKEN"]` en runtime. Los archivos del plan y del repo lo referencian **solo** como `$HF_TOKEN`; su valor no aparece en ningún archivo versionado.

**Fallo temprano (congelado).** Antes de construir o correr nada, si la lista de modelos a ejecutar incluye algún `gated` y `.env` no existe o `HF_TOKEN` está ausente/vacío, se aborta con `ValueError` en español que nombre el modelo y el archivo faltante — **nunca** a mitad de la descarga. El mensaje de error **no** imprime el valor del token.

## 5. Cross-task acceptance

Se verifica al final del ciclo (`create-test-plan` / `run-test-plan`), no dentro de ningún subtask:

1. **Regresión legacy intacta.** `pytest -q` verde y `git diff main --stat` no muestra cambios en ninguno de los archivos de F0.
2. **Barrido completo y consistente.** `data/2026/detalle_2026.csv` tiene exactamente **448** filas (14 modelos × 32 comandos), los 14 valores de `modelo` coinciden con `MODELOS_2026`, y no hay `idx` faltantes ni duplicados por modelo.
3. **Cobertura de la etapa 2.** Toda fila con `match_exact == False` en `detalle_2026.csv` tiene una fila correspondiente en `etiquetas_errores.csv` con al menos una etiqueta válida, y `categorias_comandos.csv` cubre los 32 `idx` exactamente una vez.
4. **Coherencia estricta/laxa.** Para cada modelo de `resumen_2026.json`, `exact_match_laxo_pct >= exact_match_pct`, y ambos en `[0, 100]`.
5. **Determinismo del juez.** Reejecutar la etapa 2 sobre el mismo `detalle_2026.csv` reproduce `etiquetas_errores.csv` y `categorias_comandos.csv` byte a byte.
6. **Trazabilidad de versiones.** Todo modelo con `transformers_pin != BASELINE_TRANSFORMERS` aparece con su motivo en `docker/README.md` **y** en `paper/02_reescrito/tablas/tabla5_versiones.tex` **y** se menciona en `03_metodologia.tex` y en `06_amenazas.tex`.
7. **Los dos papers compilan.** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` termina con código 0 en `paper/01_original/` y en `paper/02_reescrito/`, produciendo `main.pdf` en ambas.
8. **Envío ciego.** `python scripts/check_anonimato.py paper/02_reescrito` sale con código 0, incluyendo la revisión de metadatos del `main.pdf` ya construido.
9. **Números del paper == números de los datos.** Cada valor de las Tablas 2/3/4 del PDF reescrito proviene de los fragmentos generados por `src/generate_tex_tables.py`; regenerar los fragmentos no produce diff.
10. **Amenazas nuevas presentes.** `06_amenazas.tex` cubre explícitamente las **cuatro** amenazas de RF16.
11. **Historial limpio.** Un commit por subtask en `feat/reescritura-experimento-2026`, y `paper_cacic_LNCS_word.docx` sigue sin trackear, y `.env` sigue sin trackear (`git ls-files` no lista `.env` ni ningún `*.env`).
12. **Ningún secreto versionado.** `git grep -iE 'hf_[A-Za-z0-9]{20,}'` sale vacío (exit 1) en todo el árbol trackeado — el patrón es estricto a propósito: un token de HuggingFace es `hf_` + ~34 alfanuméricos, mientras que `hf_repo_id`, `.hf_cache`, `HF_TOKEN` y `HF_HOME` son identificadores legítimos del diseño y deben seguir existiendo. Además, `git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env' docker/` sale vacío, y `git ls-files` no lista `.env`.
