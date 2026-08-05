# Reejecución del experimento con roster 2026, verificación automática en dos etapas y reescritura del paper en LaTeX

**Goal:** Rehacer el estudio comparativo de SLMs para interpretación de comandos de domótica con un registro de 16 modelos (roster activo de 12, tras los Deltas 2026-08-04 y 05), reemplazar la clasificación manual de errores por un pipeline automático de dos etapas (coincidencia textual + juez LLM con categorías cerradas), aislar cada modelo en su propia imagen Docker, y reescribir el paper como un conjunto de archivos `.tex` LNCS listos para envío ciego a CACIC 2026.

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

### Delta 2026-08-03

# Delta 02

Tres cambios, decididos por el usuario tras el AUTH-BLOCKER de Phase 3. Los dos primeros son
decisiones suyas; el tercero es la correccion de un hecho que resulto falso.

## D2.1 — Roster de 12: los dos modelos gated salen

**Anula `03-delta-01-modelos-gated.md`.** El usuario eligio "Roster de 12, arrancar ya".

`google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct` **salen del roster activo**. El roster
del barrido queda en **12 modelos**.

Causa, verificada por root (no asumida) — y correccion de un error de root en Phase 1:

| Endpoint | gemma-3-270m-it | Llama-3.2-1B-Instruct |
|---|---|---|
| `whoami-v2` | 200 | 200 |
| `/api/models/<id>` (metadatos) | 200 | 200 |
| `/api/models/<id>/tree/main` | 200 | 200 |
| **`/<id>/resolve/main/config.json`** | **403** | **403** |

El token es valido (cuenta `LautaroL`, token `cacic-2026`, rol `read`). Lo que falta es **acceso a los
repos**: la API los marca `gated: "manual"`, o sea que el dueño aprueba las solicitudes a mano y
todavia no lo hizo.

**El error a no repetir:** root valido el acceso en Phase 1 con `/api/models/<id>` y de su 200
concluyo que las licencias estaban aceptadas. Ese endpoint solo expone metadatos y responde 200 con
cualquier token valido, sin importar el acceso a archivos. **El unico check que prueba acceso de
descarga es `/resolve/<rev>/<archivo>`.** Cualquier verificacion futura de un repo gated debe usar
ese, no el de metadatos.

Como implementarlo:

- Los dos modelos **permanecen en el registro** de `models_2026.py`, marcados `gated=True` y
  **excluidos del roster activo del barrido**, con el motivo registrado (403 en `resolve`, aprobacion
  manual pendiente al 2026-08-03). No los borres: si la aprobacion llega, reactivarlos debe ser
  cambiar un flag, no reescribir codigo.
- La plomeria de `$HF_TOKEN` y el fail-fast **se conservan** — G1 (`test_credenciales_gated.py`) sigue
  siendo un test valido del mecanismo. Lo que cambia es que ningun modelo del roster activo lo
  necesita, asi que el barrido no debe exigir `HF_TOKEN` para correr los 12.
- El invariante de F3 "exactamente dos gated" pasa a describir el **registro**, no el roster activo.
  El roster activo tiene **cero** modelos gated. Ajusta el contrato para que diga cual de las dos
  cosas cuenta, sin ambiguedad.
- El paper debe declarar la exclusion como **limitacion explicita**: dos modelos del diseño original
  quedaron fuera por acceso restringido no otorgado, no por una decision metodologica. Va en la
  seccion de modelos evaluados y en §6.

## D2.2 — Pins de transformers por modelo (revierte la premisa de `d6c7581`)

El usuario eligio "Pins por modelo".

**El commit `d6c7581` esta construido sobre una premisa falsa y hay que corregirlo.** Ese commit
aplico `transformers>=4.57.0,<5.0.0` a todo el roster argumentando que una unica version mayor
*elimina* el confusor de versiones divergentes. La evidencia de Phase 3 refuta que exista tal
version:

- **Andan con 4.57.6:** `granite-4.0-350m`, `granite-4.0-h-350m`, `SmolLM2-360M`, `LFM2.5-1.2B`.
- **Requieren 5.x:** `LFM2.5-230M`, `LFM2.5-350M`, `Qwen3.5-0.8B`.
- **`granite-4.0-350m` se rompe con 5.14.1** — `ValueError: has_previous_state can only be called on
  LinearAttention layers...`, regresion de transformers 5.x en el cache hibrido de Granite 4.

Es decir: 4.57.6 y 5.x son ambos necesarios y mutuamente excluyentes sobre el roster. Log de sondeo
en `.claude-scratch/logs/probe.log`.

Que hacer:

- Volver a **un pin por modelo**, con la version resuelta y **el motivo** de cada divergencia
  registrados en la matriz de `docker/README.md` y en el sitio del pin en `models_2026.py`. Este es el
  requisito original del usuario: *"Documenta si usas versiones distintas de la libreria por temas de
  compatibilidad"*.
- **Completar la matriz para los 12 modelos del roster activo**, no solo para los 7 ya sondeados. Los
  5 restantes hay que probarlos; no infieras su version por familia.
- Distinguir en la documentacion los pins **necesarios** (una version distinta o el modelo no corre)
  de los meramente **heredados** del baseline. Solo los primeros son confusores reales.
- **El confusor vuelve**, asi que ahora si hay que declararlo: versiones divergentes de la libreria de
  inferencia entre modelos afectan la comparabilidad de la latencia. Va en metodologia y en §6
  Amenazas a la Validez, redactado sin datos identificatorios.
- Revisa la validez de `data/2026/detalle/granite-4-0-350m.csv`: se produjo bajo 4.57.6. Si ese
  termina siendo su pin definitivo, el CSV sigue valido y no hay que recomputarlo. Si cambia, se
  descarta y se rehace. **Todo CSV debe corresponder a la version efectivamente fijada para ese
  modelo** — ese es el invariante que importa.

## D2.3 — Los LFM2.5 base **si** tienen chat template (correccion de hecho)

Hallazgo de Phase 3: `LFM2.5-230M` y `LFM2.5-350M` resuelven a `chat_template`, **no** a
`raw_completion`.

Esto **falsea la premisa D2 del diagnostico inicial** y, con ella, parte de §A1 de `02-interview.md`,
que justificaba la ruta de raw-completion precisamente por la ausencia de chat template en esos dos
modelos. La tabla de §2.1 necesita correccion.

Consecuencias:

- Corregir la tabla de §2.1 y cualquier texto del plan que afirme que esos dos modelos van por
  raw-completion.
- La **ruta de raw-completion en si no se elimina**: sigue siendo el fallback correcto por capacidad
  (`tokenizer.chat_template is None`) y esta implementada y testeada. Simplemente **ningun modelo del
  roster actual la ejercita**. Decidir y documentar si queda como codigo de fallback no ejercitado o
  si se cubre solo con un test sintetico — lo que no puede pasar es que el paper afirme que dos
  modelos se prompted por raw completion cuando no fue asi.
- Sacar del paper la salvedad de comparabilidad que se iba a declarar por esos dos modelos: ya no
  aplica, porque los 12 del roster usan chat template. Es una simplificacion, no una omision.

## Invariantes que no cambian

F0 congelado (`data/resultados_experimento_*`, `tests/test_metricas.py` byte-identicos y verdes);
salidas nuevas bajo `data/2026/`; `--cpus=2 --memory=8g` CPU-only en host ocioso; temperatura 0;
7 etiquetas de taxonomia con exactitud estricta y laxa; envio ciego sin datos identificatorios;
tablas y figuras generadas por script; el token solo en `.env` y `git grep -iE 'hf_[A-Za-z0-9]{20,}'`
vacio; artefactos del paper estrictamente aguas abajo del barrido y del juez.

### Delta 2026-08-04

# Delta 03

Decisiones del usuario del 2026-08-03, tomadas tras el cierre del Delta 02. **El usuario declaro que
no va a estar disponible y pidio explicitamente: "si en un momento dudas, deci lo que sea mejor y
documentalo".** Toda ambiguedad que aparezca de aca en adelante se resuelve con la mejor decision
tecnica disponible y se documenta en el commit y en el paper; **no se bloquea esperando respuesta**.

## D3.1 — `Qwen3.5-2B` sale; entra `Qwen/Qwen2.5-1.5B-Instruct`

Palabras del usuario: *"Ignoralo y usa el otro Qwen. Ademas, usa Qwen-2.5-1.7-instruct"*.

- **`Qwen3.5-2B` sale del roster activo, sin condicion.** No hay que sondearlo bajo 5.14.1 y **el gate
  bloqueante de `TODO_05` queda eliminado** — era precisamente lo que iba a detener el barrido a mitad
  de camino. Se conserva en el registro con el motivo: falla bajo 4.57.6 (`ValueError`, "upgrade
  transformers"), su confirmacion positiva bajo 5.14.1 nunca se obtuvo, y el usuario decidio
  excluirlo en lugar de perseguirla. El "otro Qwen" que queda es `Qwen3.5-0.8B`, ya confirmado
  `PROBE_OK` en 5.14.1.
- **Entra `Qwen/Qwen2.5-1.5B-Instruct`.** Correccion de ID verificada por root: el usuario escribio
  "Qwen-2.5-1.7-instruct", que **no existe** (`Qwen/Qwen2.5-1.7B-Instruct` → HTTP 404). El modelo real
  es `Qwen/Qwen2.5-1.5B-Instruct` (HTTP 200), y es uno de los dos Qwen2.5 del paper original, asi que
  el usuario lo confirmo explicitamente al elegirlo entre las opciones.
- **Beneficio metodologico que hay que aprovechar en el paper:** este modelo estaba en el paper
  original, asi que su reincorporacion **restaura una linea de continuidad con la Tabla 2 publicada**
  que el Delta 01 habia perdido al sacar los Qwen2.5. Permite comparar directamente el resultado nuevo
  contra el publicado para un mismo modelo, lo cual es una validacion cruzada del harness, no solo un
  dato mas. Decirlo en la seccion de modelos evaluados.
- **Su grupo de version hay que sondearlo, no inferirlo.** Qwen2.5 es una familia madura y lo esperable
  es grupo A (4.57.6), pero inferir por familia es exactamente el error que oculto el fallo de
  Granite 4. Sondear y registrar la evidencia antes de meterlo al barrido.
- **Roster activo final: 12 modelos.** Verificar que los invariantes de conteo (F3 y los checks del
  TEST_PLAN que cuentan 12/roster) sigan cuadrando tras el intercambio, y ajustar el que haga falta.

## D3.2 — Ejecucion secuencial con pinning de nucleos

El usuario pregunto si Docker admite atribucion de recursos "hard" para correr varios modelos
concurrentemente. Root respondio con la precision tecnica y el usuario eligio **secuencial**.

Lo que hay que implementar y documentar:

- **Un modelo a la vez**, con `--cpuset-cpus` fijando el contenedor a un par de nucleos concreto,
  ademas de `--cpus=2 --memory=8g`. El pinning hace la asignacion determinista entre corridas.
- **Registrar el razonamiento en la metodologia del paper**, porque es una decision metodologica
  defendible y un revisor puede preguntarla: `--cpus` es una **cuota** de CFS, no una reserva;
  `--memory` es un **techo**, no una reserva (Docker no ofrece reserva dura de RAM, solo
  `--memory-reservation`, que es blanda). `--cpuset-cpus` si fija nucleos y elimina la competencia por
  tiempo de CPU. **Pero la inferencia de LLM en CPU esta limitada por ancho de banda de memoria, y la
  cache L3 y el bus de memoria no se pueden particionar por contenedor.** Por eso correr 4 modelos en
  paralelo habria inflado los tiempos por comando de forma invisible en la tabla final. Se eligio
  secuencial para que la latencia siga siendo comparable con el paper original, aun a costa de ~7 h de
  reloj en lugar de ~2 h.
- Exactitud, taxonomia y equivalencia semantica **no** dependen de esto (temperatura 0, deterministas).
  Solo la latencia lo hacia.

## D3.3 — Persistencia incremental: anotar cada resultado al obtenerlo

Instruccion textual del usuario: *"Recuerda anotar cada resultado cuando lo obtengas asi no perdes
informacion si hay un error a mitad de las pruebas"*.

Esta sesion ya perdio trabajo tres veces (reinicio del backend de Docker, salida del proceso, error
529), asi que esto no es teorico:

- El CSV de cada modelo se escribe **en cuanto ese modelo termina**, no al final del barrido.
- **Commit inmediato por modelo**, con el nombre del modelo y su version de `transformers` en el
  mensaje. Nada de acumular 12 resultados sin versionar: lo que solo existe en el working tree es
  exactamente lo que cuesta una caida.
- Lo mismo aplica a la etapa 2 (juez): resultados por modelo o por lote persistidos y commiteados a
  medida que salen, no al cierre.
- Si un modelo falla de forma irrecuperable, **se registra el fallo con su error textual y el barrido
  continua con el siguiente**. No se bloquea el barrido entero por un modelo, y la exclusion se
  documenta en el paper como limitacion. El usuario no esta disponible para desbloquear.

## D3.4 — Sub-agentes por tarea

Instruccion textual del usuario: *"usa sub-agentes para cada tarea asi mantenes tu contexto limpio"*.

Ya es el patron del ciclo y se mantiene: cada subtask se implementa en su propio sub-agente, con el
modelo mas barato que sea capaz de la tarea, pasando artefactos por disco y reportando poco a la raiz.
Confirmado como requisito explicito del usuario, no solo como preferencia de la raiz.

## Invariantes que no cambian

F0 congelado (`data/resultados_experimento_*`, `tests/test_metricas.py` byte-identicos y verdes);
salidas nuevas bajo `data/2026/`; CPU-only sobre host ocioso; temperatura 0; 7 etiquetas de taxonomia
con exactitud estricta y laxa; el juez es el mejor de la etapa 1 con el sesgo documentado como amenaza;
envio ciego sin datos identificatorios; tablas y figuras generadas por script; el token solo en `.env`
y `git grep -iE 'hf_[A-Za-z0-9]{20,}'` vacio; los dos modelos gated siguen fuera del roster activo y
en el registro tras su flag; artefactos del paper estrictamente aguas abajo del barrido y del juez.

## 2. Formalized specification

### 2.1 Registro de 15 modelos y roster activo de 12

`Qwen2.5-0.5B-Instruct` se **elimina** (superado por `Qwen3.5-0.8B`). `Qwen2.5-1.5B-Instruct` **vuelve al roster activo** por el Delta 2026-08-04 (ver la nota de continuidad más abajo). `google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct` **quedan en el registro pero fuera del roster activo** porque el acceso de descarga no fue otorgado (`403` en `/<id>/resolve/main/config.json` con un `$HF_TOKEN` válido; los repos están marcados `gated: "manual"` y la aprobación manual seguía pendiente al 2026-08-03). `Qwen3.5-2B` **queda en el registro pero fuera del roster activo** por decisión del usuario del 2026-08-04 (ver más abajo). `SmolLM2` aporta los dos tamaños del modelo del paper original. **El único endpoint que prueba acceso de descarga es `/<id>/resolve/<rev>/<archivo>`**; `/api/models/<id>` responde 200 con cualquier token válido y no prueba nada — ese fue el error de verificación de Phase 1 y no debe repetirse.

Tabla (15 filas; columnas `activo` y `grupo de versión` agregadas por el Delta 2026-08-03; el `prompting` de las dos filas LFM2.5 base ya está corregido a `chat_template`):

| # | `nombre` | `hf_repo_id` | `params_b` | `tier` | prompting | `gated` | `activo` | grupo |
|---|----------|--------------|-----------|--------|-----------|---------|----------|-------|
| 1 | `LFM2.5-230M` | `LiquidAI/LFM2.5-230M` | 0.23 | sub-1B | chat_template | no | sí | B (5.x) |
| 2 | `LFM2.5-350M` | `LiquidAI/LFM2.5-350M` | 0.35 | sub-1B | chat_template | no | sí | B (5.x) |
| 3 | `granite-4.0-350m` | `ibm-granite/granite-4.0-350m` | 0.35 | sub-1B | chat_template | no | sí | A (4.57.x) |
| 4 | `granite-4.0-h-350m` | `ibm-granite/granite-4.0-h-350m` | 0.34 | sub-1B | chat_template | no | sí | A (4.57.x) |
| 5 | `Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | 0.8 | sub-1B | chat_template | no | sí | B (5.x) |
| 6 | `SmolLM2-360M-Instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36 | sub-1B | chat_template | no | sí | A (4.57.x) |
| 7 | `gemma-3-270m-it` | `google/gemma-3-270m-it` | 0.27 | sub-1B | chat_template | sí | **no** | — |
| 8 | `LFM2.5-1.2B-Instruct` | `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2 | 1-2B | chat_template | no | sí | A (4.57.x) |
| 9 | `granite-4.0-1b` | `ibm-granite/granite-4.0-1b` | 1.6 | 1-2B | chat_template | no | sí | A (4.57.x) |
| 10 | `granite-4.0-h-1b` | `ibm-granite/granite-4.0-h-1b` | 1.5 | 1-2B | chat_template | no | sí | A (4.57.x) |
| 11 | `Qwen3.5-2B` | `Qwen/Qwen3.5-2B` | 2.0 | 1-2B | chat_template | no | **no** | B (5.x) |
| 12 | `OLMo-2-0425-1B-Instruct` | `allenai/OLMo-2-0425-1B-Instruct` | 1.0 | 1-2B | chat_template | no | sí | A (4.57.x) |
| 13 | `SmolLM2-1.7B-Instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71 | 1-2B | chat_template | no | sí | A (4.57.x) |
| 14 | `Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` | 1.0 | 1-2B | chat_template | sí | **no** | — |
| 15 | `Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54 | 1-2B | chat_template | no | sí | A (4.57.x) |

Roster activo: **12 modelos, 6 por tier** (sub-1B: 1, 2, 3, 4, 5, 6; 1-2B: 8, 9, 10, 12, 13, 15). Cero modelos gated en el roster activo. Registro completo: **15**; excluidos: **3** (7, 11, 14).

**`Qwen3.5-2B` — exclusión del 2026-08-04.** Sale del roster activo **sin condición**, por decisión del usuario. Motivo registrado: falla bajo 4.57.6 con `ValueError` que pide `pip install --upgrade transformers` (evidencia negativa propia, `.claude-scratch/logs/probe.log`), y su **confirmación positiva bajo 5.14.1 nunca se obtuvo** porque la sonda no llegó a correr (el almacenamiento del daemon de Docker quedó en modo solo lectura a mitad de ronda — bloqueo de infraestructura, **no** evidencia sobre el modelo). El usuario eligió excluirlo en lugar de perseguir esa confirmación. **Redacción obligatoria en todo el plan y en el paper:** `Qwen3.5-2B` queda fuera por **compatibilidad no verificada**, nunca por "incompatibilidad demostrada con las dos versiones mayores" — no hay evidencia de que falle bajo 5.14.1, solo ausencia de evidencia de que funcione. Permanece en el registro con `activo=False` y `motivo_exclusion` no vacío; reactivarlo es un cambio de flag más su sonda positiva, nunca una reescritura de código. **El gate bloqueante que el subtask 05 tenía por este modelo queda eliminado**: habría detenido el barrido a mitad de camino sin nadie disponible para desbloquearlo.

**`Qwen2.5-1.5B-Instruct` — reingreso del 2026-08-04, con corrección de ID.** El usuario escribió *"Qwen-2.5-1.7-instruct"*. Ese ID **no existe**: `Qwen/Qwen2.5-1.7B-Instruct` devuelve HTTP 404. El modelo real es **`Qwen/Qwen2.5-1.5B-Instruct`** (HTTP 200), que es uno de los dos Qwen2.5 del paper original, y el usuario lo confirmó al elegirlo entre las opciones. **Nadie puede reintroducir el ID `Qwen/Qwen2.5-1.7B-Instruct` en ningún archivo**: no es un modelo.

Su **grupo de versión es A (`transformers` 4.57.6)**, y lo es **por evidencia propia, no por inferencia de familia**. Sonda ejecutada el 2026-08-04 dentro de la imagen ya construida del grupo A (`slm-domotica-2026:granite-4-0-350m`, con `transformers==4.57.6` confirmado dentro del contenedor), con una copia adaptada del script de sondeo (`.claude-scratch/probe_qwen25_15b.py`, idéntica en procedimiento salvo que el `hf_repo_id` va hardcodeado porque el modelo todavía no estaba en el registro). Salida verbatim (`.claude-scratch/logs/probe_qwen25_15b_groupA.log`):

```
PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template
```

Carga real del modelo + `apply_chat_template` funcionando + `generate()` greedy. **Fuerza de la evidencia: fuerte pero parcial**, exactamente el mismo estándar que las filas *heredadas* del grupo A: confirma que 4.57.6 funciona; no prueba ni descarta que 5.14.1 también funcionaría (esa sonda no se corrió). Por lo tanto su pin es **heredado** (`motivo_pin == ""`), no necesario. El plan **nunca** puede afirmar que su grupo se dedujo de la familia Qwen: la propia matriz muestra que Qwen3.5 resuelve a grupo B, así que "familia Qwen" no predice versión. Detalle completo en `.claude-scratch/20260802-1200-cacic-paper-rewrite/07-evidencia-versiones.md`.

**Continuidad con la Tabla 2 publicada (aporte metodológico, no contabilidad).** Con este reingreso, los modelos presentes **en los dos estudios** —el publicado y el 2026— pasan de **2 a 3**:

| modelo (en ambos estudios) | `params_b` | estricta publicada (%) | JSON válido publicado (%) | latencia promedio publicada (s) |
|---|---|---|---|---|
| `SmolLM2-360M-Instruct` | 0.36 | 18.8 | 100.0 | 15.617 |
| `Qwen2.5-1.5B-Instruct` | 1.54 | 50.0 | 100.0 | 43.159 |
| `SmolLM2-1.7B-Instruct` | 1.71 | 59.4 | 100.0 | 61.869 |

Fuente única de esas cifras: `data/resultados_experimento_resumen.json` (archivo **F0**, inmutable). Ninguna se escribe a mano en el paper: entran por las columnas generadas de la Tabla 2 (ver F8).

Con el modelo **fijo**, cualquier diferencia entre el número nuevo y el publicado es atribuible al **protocolo y al harness** (cláusula taxativa de RF1, versión de `transformers`, fecha y carga del host), no al modelo. Eso convierte a estas tres filas en una **validación cruzada del harness nuevo contra un resultado publicado**, y en la única forma de **cuantificar** el efecto del endurecimiento del prompt en vez de solo afirmarlo. `Qwen2.5-1.5B-Instruct` es además el único ancla del tier 1–2B con familia distinta de SmolLM2, así que las tres cubren los dos tiers y dos familias.

**Precisión obligatoria (no sobreafirmar).** §3.3 ya declara que la exactitud estricta **no es directamente comparable** con la del paper original porque el prompt se endureció; y la latencia arrastra además el confusor de los dos grupos de versión. La comparación es por lo tanto una **validación cruzada con el modelo controlado**, no una réplica: se reporta, se interpreta y, si hay divergencia, **se discute** — nunca se entierra ni se presenta como replicación exacta.

El `modo_prompting` de la tabla es la **expectativa**; el código lo decide en runtime por capacidad (`tokenizer.chat_template is None`), nunca por ID hardcodeado. Si la detección discrepa de la tabla, gana la detección y se corrige la tabla del paper. La detección de Phase 3 confirmó `chat_template` en los 11 modelos del roster activo que ya estaban sondeados entonces, incluidos los dos LFM2.5 base, que el diagnóstico inicial suponía sin chat template; la sonda del 2026-08-04 confirmó `chat_template` también en `Qwen2.5-1.5B-Instruct` (modo reportado en la línea `PROBE_OK` de arriba). Los **12** del roster activo resuelven a `chat_template`. La tabla ya refleja la detección.

### 2.2 Requisitos funcionales

- **RF1** — Prompt de sistema 2026: el prompt del paper más una cláusula **taxativa** que declare que los valores entre paréntesis son los únicos outputs aceptados textualmente y que usar sinónimos es una violación de formato (texto exacto congelado en §4, F2).
- **RF2** — Ruta de prompting por *raw completion* para tokenizers sin `chat_template`, detectada por capacidad. Ningún modelo del roster activo la ejercita (los 12 resuelven a `chat_template`). La ruta se conserva como fallback por capacidad y su cobertura es un test sintético con un tokenizer sin `chat_template` (`tests/test_prompt_2026.py`), no un modelo del roster. Ni el plan ni el paper pueden afirmar que algún modelo se prompteó por raw completion.
- **RF3** — Barrido **reanudable por modelo**: un CSV por modelo; reejecutar salta los modelos ya completos salvo `--force`.
- **RF4** — **Una imagen Docker por modelo** — **12 imágenes activas**, una por modelo del roster activo, cada una con el pin de su grupo de versión. Las **tres** imágenes de los modelos excluidos siguen **definidas** (misma receta, incluida la plomería `--env-file` para los dos gated) pero **no se construyen ni se corren**. Compartiendo un único volumen de caché HF. Ejecución **estrictamente secuencial, un modelo a la vez**, con `--memory=8g --cpus=2 --cpuset-cpus=0-1`, CPU-only (ver RNF6 para el razonamiento del pinning). Como el roster activo no tiene modelos gated, **el barrido corre sin credenciales y no debe exigir `$HF_TOKEN`**.
- **RF5** — Toda divergencia de versión respecto del baseline debe quedar documentada (qué modelo la forzó, qué error evita) en `docker/README.md`, en el sitio del pin, y en el texto de metodología y de amenazas del paper. Lo mismo para cualquier uso de `trust_remote_code`. Cada divergencia debe declararse como **necesaria** (hay evidencia empírica de que el modelo no corre con la otra versión mayor) o **heredada** (el modelo corre con el pin de su grupo y no se probó la alternativa). Solo las necesarias son confusores reales; ambas se documentan.
- **RF6** — Etapa 1 (automática): coincidencia textual exacta campo a campo, como hoy.
- **RF7** — Selección determinista del juez: mayor `exact_match_pct` de la etapa 1; empate → mayor `params_b`; empate persistente → orden del roster. Decodificación greedy/temperatura 0 en etapa 1 y etapa 2.
- **RF8** — Etapa 2, trabajo A: el juez clasifica cada respuesta con `match_exact == False` en un subconjunto **no vacío** de las 7 etiquetas cerradas (§4, F4). Salida cerrada, no abierta.
- **RF9** — Etapa 2, trabajo B: el juez clasifica cada uno de los 32 comandos en **exactamente una** de las 6 categorías lingüísticas cerradas (§4, F4). Reemplaza el etiquetado manual de la Tabla 3; **no** se agrega columna al dataset.
- **RF10** — Dos métricas titulares: **exactitud estricta** (`match_exact`) y **exactitud laxa** (`match_exact` o etiqueta en `{sin_error_semantico, uso_de_sinonimos}`). Ambas en la Tabla 2; la brecha se discute como resultado.
- **RF11** — Tabla 4 con las **5 categorías publicadas** (se separa `alucinacion_valor_unidad` de `valor_numerico_incorrecto`; el legacy las fusionaba) más las 2 etiquetas nuevas.
- **RF12** — Figuras rediseñadas: **barras horizontales agrupadas por tier**, nombres largos legibles, sin apiñamiento a 12 modelos.
- **RF13** — Tablas 1–5 del paper generadas por script desde los JSON de resultados hacia fragmentos `.tex` que `main.tex` hace `\input`.
- **RF14** — Dos árboles LaTeX: `paper/01_original/` (transcripción fiel del `.docx`) y `paper/02_reescrito/`, cada uno con `main.tex`, un `.tex` por sección y `refs.bib`, autocontenidos (`llncs.cls` y `splncs03.bst` copiados), y ambos compilando a PDF.
- **RF15** — **Envío ciego**: cero datos identificatorios en `paper/02_reescrito/` ni en su PDF (sin autores, afiliaciones, agradecimientos, financiamiento, URL del repositorio, ORCID, ni metadatos identificatorios). Verificado por script, no por inspección.
- **RF16** — El paper reescrito debe incorporar **tres** amenazas a la validez **nuevas** (exactamente tres; no se agrega una cuarta): (1) sesgo de auto-favorecimiento del juez; (2) **versiones divergentes de la librería de inferencia entre modelos** (dos grupos de versión mutuamente excluyentes sobre el roster) como confusor de la latencia, **incluyendo** que un modelo registrado (`Qwen3.5-2B`) quedó fuera del roster evaluado porque su compatibilidad de versión no se pudo verificar — evidencia adicional de que el confusor no es teórico; (3) exclusión de modelos del registro por causas ajenas al método: **dos** por acceso restringido no otorgado y **uno** por compatibilidad de versión no verificada, **tres sobre quince** en total. **Se elimina** la amenaza de incomparabilidad de los dos modelos base prompteados por raw completion: ya no aplica, porque los 12 del roster activo usan chat template. Es una simplificación, no una omisión.
- **RF17** — **Credenciales**: el token de HuggingFace vive solo en `.env` (ignorado por git) y se pasa a los dos contenedores *gated* con `docker run --env-file .env`. Nunca como `ARG`/`ENV` de Dockerfile, `--build-arg`, `COPY`, capa de imagen ni `login` interactivo. Su **valor** no puede aparecer en ningún archivo versionado. Si falta, el barrido falla **temprano** con un mensaje claro en español que no imprime el token. Ningún modelo del roster activo lo necesita; el fail-fast se dispara solo si alguien reactiva un modelo excluido.
- **RF18** — **Reproducibilidad declarada**: el paper reescrito debe declarar, en la sección de metodología/reproducibilidad y en §6 Amenazas a la Validez, que **3 de los 15 modelos del registro 2026 quedaron fuera del barrido** por causas ajenas al método —**2** (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) por acceso de descarga no otorgado y **1** (`Qwen3.5-2B`) por compatibilidad de versión **no verificada**—, que los **12 evaluados son todos de repositorios públicos** y que por lo tanto reproducir el barrido publicado **no requiere token ni aceptar licencias**. Redactado sin datos identificatorios (RF15).
- **RF19** — **Continuidad con la Tabla 2 publicada (validación cruzada del harness).** Los **tres** modelos presentes en los dos estudios (`SmolLM2-360M-Instruct`, `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`) se comparan explícitamente contra sus cifras publicadas. Concretamente: (a) la sección de **modelos evaluados** debe decir que `Qwen2.5-1.5B-Instruct` se reincorporó **deliberadamente** para sostener esa línea de continuidad, y que con él los anclas publicados pasan de 2 a 3, cubriendo los dos tiers y dos familias; (b) la sección de **resultados** debe **reportar** la comparación —exactitud estricta nueva vs. publicada y latencia promedio nueva vs. publicada, por las columnas generadas de la Tabla 2 (F8), nunca a mano—; (c) la **discusión** debe interpretar la comparación y, si hay divergencia, **discutirla explícitamente** en vez de enterrarla, atribuyéndola al protocolo/harness (modelo controlado) y no al modelo; (d) el texto debe declarar la salvedad de que **no es una réplica**: el prompt se endureció (RF1), así que la estricta no es directamente comparable, y la latencia arrastra el confusor de los dos grupos de versión. Fuente única de las cifras publicadas: `data/resultados_experimento_resumen.json` (F0, inmutable).
  **Retractado (Delta 04).** No hay un "paper 2025" publicado: era un borrador de este mismo trabajo,
  nunca publicado ni a publicarse, y sus cifras se produjeron con el entorno sin fijar
  (`transformers`, `torch`, `attn_implementation`, `revision`), por lo que no son comparables de forma
  válida — presentarlas como columnas de continuidad invitaba a una comparación inválida. Las
  columnas de continuidad de la Tabla 2 y toda la prosa de "ronda inicial"/continuidad se eliminaron
  del paper (no solo su atribución). La versión legítima de esta idea, que sí se mantiene, es la
  comparación generacional: `Qwen2.5-1.5B-Instruct`, representante de la generación anterior de SLMs,
  medido ahora bajo el mismo protocolo 2026 que el resto del roster. Ver ítem 15 revisado de
  §5 Cross-task acceptance y Check C15 de `TEST_PLAN.md`. `data/resultados_experimento_resumen.json`
  sigue existiendo como F0 congelado, pero ya no se lee desde `src/generate_tex_tables.py`.
- **RF20** — **Persistencia incremental y continuación ante fallo.** (a) El CSV de cada modelo se escribe en cuanto **ese** modelo termina, y se **commitea de inmediato**, con el nombre del modelo y su versión efectiva de `transformers` en el mensaje. (b) Lo mismo en la etapa 2: los CSV del juez se persisten por lote (por modelo en el trabajo A, por comando en el trabajo B) y se commitean a medida que salen, no al cierre. (c) Si un modelo falla de forma irrecuperable, se registra el fallo **con su error textual**, el barrido **continúa con el siguiente**, y la exclusión resultante se documenta en el paper como limitación. **El barrido no se bloquea nunca por un modelo**: el usuario no está disponible para desbloquearlo. Contrato ejecutable en F12.

### 2.3 Requisitos no funcionales

- **RNF1** — Hardware: la misma máquina Windows con Docker Desktop, CPU-only, `--memory=8g --cpus=2 --cpuset-cpus=0-1`, sobre host ocioso, para preservar la comparabilidad de la tabla de latencia con el paper original. Sin GPU. El valor de `--cpuset-cpus` es **el mismo para las 12 corridas y para el juez**.
- **RNF2** — Presupuesto: barrido **secuencial** de ~7 h de reloj (12 modelos de a uno) y ~20–40 GB de descargas. Debe tolerar interrupciones (RF3) y continuar ante fallos individuales (RF20).
- **RNF3** — Determinismo: `do_sample=False`, `temperature=None`, `top_p=None` en las dos etapas.
- **RNF4** — Los tests existentes (`pytest -q`) quedan verdes en todo momento, en cada commit.
- **RNF5** — Un commit por subtask, en la rama `feat/reescritura-experimento-2026` (ya creada desde `main`). **Excepción explícita del Delta 2026-08-04:** el subtask 05 commitea **un commit por modelo** (RF20a), y el 08 uno por lote; siguen siendo "un subtask, una unidad de trabajo", con granularidad de commit más fina a propósito.
- **RNF6** — **Ejecución secuencial con pinning de núcleos, y su razonamiento.** Un modelo a la vez, con el contenedor fijado a un par de núcleos concreto vía `--cpuset-cpus`. El razonamiento **debe llegar a la metodología del paper**, porque es una decisión metodológica defendible que un revisor puede cuestionar: `--cpus` es una **cuota** del planificador CFS, no una reserva; `--memory` es un **techo**, no una reserva (Docker solo ofrece `--memory-reservation`, que es blanda); `--cpuset-cpus` **sí** fija núcleos y elimina la competencia por tiempo de CPU. **Pero la inferencia de LLM en CPU está limitada por ancho de banda de memoria, y ni la caché L3 ni el bus de memoria se pueden particionar por contenedor**, así que correr 4 modelos en paralelo habría inflado los tiempos por comando de forma **invisible** en la tabla final. Se eligió secuencial para que la latencia siga siendo comparable con el paper original, al costo de ~7 h de reloj en lugar de ~2 h. Exactitud, taxonomía y equivalencia semántica **no** dependen de esta decisión (temperatura 0, deterministas): solo la latencia lo hacía.
- **RNF7** — **Sub-agentes por tarea (requisito explícito del usuario, no preferencia de la raíz).** Cada subtask se implementa en su propio sub-agente, con el modelo más barato que sea capaz de la tarea, pasando artefactos por disco y reportando poco a la raíz, para mantener limpio el contexto del orquestador.

### 2.4 Fuera de alcance (out of scope)

- Cualquier flujo de login interactivo (`huggingface-cli login`, `huggingface_hub.login()`) o gestión de credenciales más allá de leer `$HF_TOKEN` del entorno. Rotar, emitir o almacenar tokens tampoco es parte del plan.
- `Qwen2.5-0.5B-Instruct` del paper original (descartado, superado por `Qwen3.5-0.8B`). **`Qwen2.5-1.5B-Instruct` sí está en alcance**: el Delta 2026-08-04 lo reincorporó al roster activo (§2.1, RF19).
- Perseguir la confirmación positiva de `Qwen3.5-2B` bajo `transformers` 5.14.1. El usuario decidió excluirlo en vez de sondearlo; sondearlo ahora no es trabajo de este plan. Queda en el registro con `activo=False`.
- Ejecución **concurrente** de varios modelos (varios contenedores a la vez), y cualquier intento de reservar ancho de banda de memoria o particionar caché L3 por contenedor: Docker no lo ofrece y RNF6 documenta por qué se eligió secuencial.
- Modificar `data/resultados_experimento_detalle.csv`, `data/resultados_experimento_resumen.json`, `tests/test_metricas.py`, `data/dataset_comandos_domotica.csv` o los módulos legacy de `src/`.
- Abstract en inglés (el paper es solo en español).
- Chequeo automático del límite de 10 páginas.
- Commitear `paper_cacic_LNCS_word.docx` (queda *untracked*, en `.gitignore`).
- Ampliar, reequilibrar o reetiquetar a mano el dataset de 32 comandos.
- Uso de GPU, cuantización, o cualquier juez externo vía API.
- `/split-todo`: es un único DAG en una única rama.

## 3. Subtask DAG

| id | title | depends_on | file | estado |
|----|-------|------------|------|--------|
| 01 | Registro de modelos 2026 y vocabularios cerrados | [] | `TODO_01_registro-modelos-2026.md` | implementado (`37a1703`) |
| 02 | Prompt taxativo 2026 y ruta de raw completion | [01] | `TODO_02_prompt-y-raw-completion.md` | implementado (`8c9287a`) |
| 03 | Harness de barrido reanudable por modelo | [01, 02] | `TODO_03_harness-barrido-reanudable.md` | implementado (`996873a`) |
| 04 | Imágenes Docker por modelo y matriz de versiones | [01, 03] | `TODO_04_docker-por-modelo.md` | implementado (`99c13aa`, `dfb8315`, `d6c7581`) |
| 05 | Ejecución del barrido completo (12 modelos del roster activo) | [04, 16, 17] | `TODO_05_ejecucion-barrido.md` | implementado (`dba3512`…`9ae5a9e`, 12 commits `data(2026): barrido de <modelo>` uno por modelo; `a1c2252`/`c4d9593` logs; `eba28f7` fallos_barrido.json) |
| 06 | Consolidación de etapa 1 y selección del juez | [05] | `TODO_06_consolidacion-y-juez.md` | implementado (`8cc0cfb`) |
| 07 | Juez LLM: taxonomía de errores y categorías lingüísticas | [06] | `TODO_07_juez-llm.md` | implementado (`1c1b453`) |
| 08 | Ejecución de la etapa 2 | [07] | `TODO_08_ejecucion-juez.md` | implementado (9 commits `data(2026): etapa 2 parcial de etiquetas_errores` por continuación ante fallo + `3668706`, `92e906f`, `416ed71`) |
| 09 | Métricas 2026: estricta, laxa, Tablas 2/3/4 | [08, 19] | `TODO_09_metricas-2026.md` | implementado (`493fdcf`, `80b1111`, `adbe2be`, `8ba1140`) |
| 10 | Figuras horizontales agrupadas por tier | [09] | `TODO_10_figuras-2026.md` | implementado (`9651ad6`, `a347bae`, `f3f3cf9`, `b072ae2`, `fb5fb89`) |
| 11 | Generador de fragmentos `.tex` de tablas | [09] | `TODO_11_generador-tablas-tex.md` | implementado (`8183ebd`, `42a5d96`, `1b84a03`) |
| 12 | Verificador de anonimato para envío ciego | [] | `TODO_12_verificador-anonimato.md` | implementado (`c1e360d`) |
| 13 | Transcripción fiel del `.docx` a `paper/01_original/` | [] | `TODO_13_transcripcion-original.md` | implementado (`0593681`, `08ca4f4`) |
| 14 | Paper reescrito: andamiaje y secciones 1–3 | [10, 11, 12, 13, 18, 19] | `TODO_14_paper-secciones-1-3.md` | implementado (`0bf5500`, `e3c34f0`, `b8e7ac7`) |
| 15 | Paper reescrito: secciones 4–7, build y anonimato | [14, 18, 19] | `TODO_15_paper-secciones-4-7.md` | implementado (`41ba9cb` + múltiples fixes posteriores de compilación/anonimato: `7372489`, `0bb1b48`, `969d92c`, `a1242bd`, `7c1cf34`, `222320d`, `6460d6d`, `8fa6380`, `7abec19`, `be7baff`, `0227a99`, `03413d1`, `8f0497d`) |
| 16 | Roster activo de 12, pins de transformers por grupo y matriz documentada | [01, 04] | `TODO_16_roster-12-y-pins.md` | **implementado (`864c694`)** — su cola de builds de Docker se trasladó al nodo `17` (ver más abajo) |
| 17 | Intercambio de roster (sale `Qwen3.5-2B`, entra `Qwen2.5-1.5B-Instruct`), pinning de núcleos y continuación ante fallo | [01, 04, 16] | `TODO_17_intercambio-roster-y-ejecucion.md` | implementado (`cbdce86`, `2be13d8`, `771a840`) |
| 18 | Corrida de control con el prompt original sobre los tres anclajes | [05] | `TODO_18_corrida-de-control.md` | **implementado (`abce9f0`, `bd9887a`, `5938b14`)** — código y tests listos; la corrida real (`docker/run_control.py`) la lanza el orquestador después de este reporte |
| 19 | Baseline re-medido de los 4 modelos del paper original (`Qwen2.5-0.5B-Instruct` + comparación) | [05, 18] | `TODO_19_baseline-original-completo.md` | **implementado (`8fb248f`, `95d1334`, `ab58049`, `44f7fd6`)** — código, imagen y tests listos; la corrida real (`docker/run_baseline.py`, sin `--dry-run`) la lanza el orquestador después de este reporte |

**Reconciliación de bookkeeping (corregida el 2026-08-04).** La afirmación anterior de esta sección —*"ningún subtask tiene casillas tildadas"*— era **factualmente falsa** y queda corregida hacia adelante: `TODO_16_roster-12-y-pins.md` tiene **36 de 39** casillas tildadas. La convención de checkboxes **sí** se usó, en ese subtask. Regla vigente:

- La historia de git sigue siendo el registro autoritativo de **qué se implementó**.
- Las casillas de `TODO_16` son un registro válido y **más fino** que la columna `estado`: distinguen el núcleo de código (commiteado en `864c694`) de la cola de operaciones de Docker que quedó bloqueada por infraestructura.
- El núcleo de `16` está **commiteado e inmutable** (`src/models_2026.py`, `docker/build_all.py`, `docker/run_sweep.py`, `docker/README.md` y sus tres archivos de test). Cualquier cambio sobre ese material va al nodo **17**, nunca como edición del texto ya commiteado de `16`.
- No se reescribe historia por esta corrección: es documentación forward-only.

**Reconciliación de estado (pasada confirmatoria final, 2026-08-04).** La columna `estado` de la tabla de arriba marcaba `05`–`11`, `14`, `15` y `17` como `pendiente` pese a que el árbol ya tenía sus commits — un desfasaje de bookkeeping, no de código: el DAG completo (barrido de los 12, etapa 2, métricas, figuras, tablas, los dos papers) ya estaba implementado y verificado en verde por `TEST_PLAN.md` antes de esta corrección. Se actualiza la columna para que coincida con la historia de git; no se tocó ningún commit ni ningún archivo de contenido.

**Concurrencia.** `12` y `13` no tienen dependencias y sus conjuntos de archivos son disjuntos de todo lo demás y entre sí: pueden ejecutarse en cualquier momento, en paralelo con `01`–`11`. `10` y `11` dependen ambos de `09` pero tocan archivos disjuntos (`src/generate_figures_2026.py` + `figures/2026/` vs. `src/generate_tex_tables.py` + `paper/02_reescrito/tablas/`): pueden correr concurrentemente. El resto es una cadena estricta. `05` y `08` son las dos tareas de ejecución largas y bloquean todo lo que va después.

**`16` → `17`, y por qué la cola de builds se movió.** `16` es correctiva de `01`/`04`; `17` es correctiva de `16`, así que `17.depends_on` **debe** incluir `16` (edita exactamente los seis archivos que `16` dejó commiteados). Las tres casillas de Docker que quedaban abiertas en `16` (reconstruir las imágenes del grupo B, confirmar la versión efectiva por grupo, no reconstruir las del grupo A) dependen del roster **posterior** al intercambio, o sea de `17`: dejarlas en `16` habría exigido `16.depends_on ∋ 17` **y** `17.depends_on ∋ 16`, que es un **ciclo**. Por eso esa cola se traslada a `17` con los conteos corregidos (**3** imágenes del grupo B a reconstruir, no 4; **1** imagen nueva del grupo A a construir, la de `Qwen2.5-1.5B-Instruct`; **8** del grupo A que no se reconstruyen), y `16` queda cerrado con una nota que apunta a `17`. El DAG queda acíclico.

**`17` bloquea todo lo que consume el roster.** El barrido no puede correr contra el roster viejo: `05.depends_on` pasa de `[04, 16]` a `[04, 16, 17]`, y `06`–`11`, `14` y `15` heredan la dependencia por transitividad (`06←05`, `07←06`, `08←07`, `09←08`, `10←09`, `11←09`, `14←10,11`, `15←14`). Ningún nodo que lea `roster_activo()`, `MODELOS_2026` o `data/2026/**` puede ejecutarse antes de `17`.

**`19` bloquea todo lo que compara contra el paper original.** `09.depends_on` pasa de `[08]` a `[08, 19]`, y `10`/`11` heredan la dependencia por transitividad (`10←09`, `11←09`), igual que con `17`. `14.depends_on` pasa de `[10, 11, 12, 13, 18]` a `[10, 11, 12, 13, 18, 19]` y `15.depends_on` de `[14, 18]` a `[14, 18, 19]`: ningún texto de `09`, `10`, `11`, `14` o `15` puede comparar contra la Tabla 2 publicada como si fuera un baseline validado. Ver el Delta 05 para la narrativa completa.

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

BASELINE_TRANSFORMERS: str = "transformers>=4.57.0,<5.0.0"   # grupo A, resuelve 4.57.6
TRANSFORMERS_5X: str        = "transformers>=5.0.0"          # grupo B, resuelve 5.14.1

@dataclass(frozen=True)
class ModeloEvaluado2026:
    nombre: str
    hf_repo_id: str
    params_b: float
    tier: Tier
    transformers_pin: str      # BASELINE_TRANSFORMERS o TRANSFORMERS_5X
    trust_remote_code: bool
    gated: bool                # requiere licencia aceptada + $HF_TOKEN para descargarse
    motivo_pin: str            # "" si el pin es heredado; el porque si el pin es NECESARIO
    activo: bool               # True <=> forma parte del roster activo del barrido
    motivo_exclusion: str      # "" si activo; el porque si no

MODELOS_2026: list[ModeloEvaluado2026]   # los 15 de §2.1, en ese orden (registro completo)

def slug(nombre: str) -> str: ...        # minúsculas; [^a-z0-9]+ -> "-"; sin guiones al borde
def roster_activo() -> list[ModeloEvaluado2026]: ...  # los 12, en orden de registro
def gated() -> list[ModeloEvaluado2026]: ...          # los 2 del registro, en orden
def por_tier(tier: Tier) -> list[ModeloEvaluado2026]: ...  # filtra el ROSTER ACTIVO
def por_nombre(nombre: str) -> ModeloEvaluado2026: ...     # busca en el REGISTRO completo
```

`src/models_2026.py` declara el **flag** `gated`; **no** lee el token ni menciona su valor.

**Invariantes congelados, verificados por test:**

- (a) `motivo_pin != ""` **si y solo si** el pin de ese modelo es **necesario**, o sea que hay
  evidencia registrada de que el modelo no corre con la otra versión mayor. Los pines **heredados**
  llevan `motivo_pin == ""`. Corolario obligatorio: todo modelo con
  `transformers_pin != BASELINE_TRANSFORMERS` tiene `motivo_pin != ""`. *(Este invariante cambia:
  antes ataba `motivo_pin` a "difiere del baseline"; ahora lo ata a "es necesario", que es la
  propiedad que el paper tiene que declarar. `granite-4.0-350m` está en el baseline **y** tiene
  motivo, porque su pin es necesario.)*
- (b) `transformers_pin in {BASELINE_TRANSFORMERS, TRANSFORMERS_5X}` para las 15 filas: exactamente
  **dos** grupos de versión. Sobre el **roster activo**: **9** modelos en el grupo A y **3** en el
  grupo B. Sobre el **registro** completo: **11** en A (los 9 activos + los 2 gated) y **4** en B
  (los 3 activos + `Qwen3.5-2B`, excluido). *(Amendado el 2026-08-04: antes decía 8 y 4 sobre el
  roster activo; el intercambio de roster movió una fila de B a A.)*
- (c) El **registro** (`MODELOS_2026`) tiene **15** modelos y exactamente **dos** con `gated = True`:
  `google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct`. El conteo de gated es una propiedad
  del **registro**, no del roster activo, y **no cambió** con el Delta 2026-08-04: `Qwen3.5-2B` está
  excluido pero **no** es gated.
- (d) El **roster activo** (`activo = True`) tiene exactamente **12** modelos, **seis por tier**, y
  **cero** con `gated = True`. El conteo del roster activo **no cambió** con el intercambio: salió
  `Qwen3.5-2B` (tier `1-2B`) y entró `Qwen2.5-1.5B-Instruct` (tier `1-2B`), así que la partición
  6 / 6 por tier se conserva.
- (e) `activo == False` **si y solo si** `motivo_exclusion != ""`. Los **tres** excluidos son
  `google/gemma-3-270m-it`, `meta-llama/Llama-3.2-1B-Instruct` y `Qwen/Qwen3.5-2B`. Reactivar un
  modelo excluido es poner `activo=True` y vaciar `motivo_exclusion`: un cambio de flag, nunca una
  reescritura de código.
- (f) `src/models_2026.py` declara el flag `gated`; **no** lee el token ni menciona su valor
  (sin cambios).
- (g) **Nuevo (2026-08-04) — separación entre exclusión y pin.** `motivo_exclusion` y `motivo_pin`
  son ortogonales: `Qwen3.5-2B` tiene **ambos** no vacíos (su pin del grupo B sigue siendo
  **necesario**, por su evidencia negativa propia bajo 4.57.6, y además está excluido). Por lo tanto
  el conjunto de pines **necesarios** sigue siendo el mismo cinco sobre el **registro**
  —`{LFM2.5-230M, LFM2.5-350M, Qwen3.5-0.8B, Qwen3.5-2B, granite-4.0-350m}`— pero sobre el
  **roster activo** son **cuatro**: `{LFM2.5-230M, LFM2.5-350M, Qwen3.5-0.8B, granite-4.0-350m}`.
  Todo test que afirme el conjunto de necesarios debe declarar sobre cuál de los dos conjuntos cuenta.
- (h) **Nuevo (2026-08-04) — IDs prohibidos.** `hf_repo_id` no puede tomar el valor
  `Qwen/Qwen2.5-1.7B-Instruct`: **ese modelo no existe** (HTTP 404). El ID correcto del modelo que
  el usuario pidió es `Qwen/Qwen2.5-1.5B-Instruct` (HTTP 200). `Qwen/Qwen2.5-0.5B-Instruct` sigue
  **descartado** del registro. Un test verifica los dos hechos para que nadie reintroduzca el ID malo.
- (i) **Nuevo (2026-08-04) — evidencia obligatoria de grupo.** Ninguna fila puede tener su
  `transformers_pin` fijado por inferencia de familia. `Qwen2.5-1.5B-Instruct` está en el grupo A por
  sonda propia (`PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template`, ver §2.1), con la misma fuerza
  de evidencia que las filas heredadas del grupo A. Su `motivo_pin` es `""` porque su pin es
  **heredado**, no necesario.

**Nota de versiones (reemplaza la nota de `d6c7581`).** La premisa de que existe una única versión
mayor de `transformers` que sirve para todo el roster es **falsa**, y la evidencia de Phase 3 la
refuta en las dos direcciones: `granite-4.0-350m` falla bajo 5.14.1 con

```
ValueError: `has_previous_state` can only be called on LinearAttention layers, and the current Cache seem to only contain Attention layers.
```

—una regresión de `transformers` 5.x en el manejo de caché híbrida/linear-attention que afecta la
arquitectura Granite 4— y `Qwen3.5-*` / `LFM2.5-{230M,350M}` fallan bajo 4.57.6 con errores de
tokenizer/versión mínima no satisfecha. 4.57.x y 5.x son **ambas necesarias y mutuamente
excluyentes** sobre el roster. El pin vuelve a ser por modelo, agrupado en dos grupos de versión,
con la matriz de §7 como fuente única. El resolutivo de grupo B (5.14.1) es precisamente la versión
que rompe `granite-4.0-350m`: es la evidencia más limpia posible de que ninguna versión mayor única
cubre el roster. La premisa de `d6c7581` (que una única versión mayor elimina el confusor) queda
por lo tanto refutada, y el confusor "versiones de librería divergentes entre modelos" se declara,
no se descarta.

**Confirmación positiva bajo 5.14.1, por modelo (actualizado al 2026-08-04, ver
`.claude-scratch/logs/probe5x.log`).** Los **3** modelos del grupo B que están en el roster activo
—`LFM2.5-230M`, `LFM2.5-350M`, `Qwen3.5-0.8B`— tienen `PROBE_OK` confirmado bajo 5.14.1, modo
`chat_template`. La confirmación positiva de `Qwen3.5-2B` bajo 5.14.1 **nunca se obtuvo** —su sonda no
llegó a correr porque el almacenamiento del daemon de Docker quedó en modo solo lectura a mitad de
ronda, un bloqueo de infraestructura y **no** evidencia sobre el modelo— y el Delta 2026-08-04 decidió
**excluirlo del roster activo en lugar de perseguirla**. `Qwen3.5-2B` conserva su pin del grupo B
dentro del registro: su evidencia negativa propia bajo 4.57.6 existe y por sí sola hace necesario el
grupo B para él. **Redacción taxativa:** no hay evidencia de que `Qwen3.5-2B` falle bajo 5.14.1; hay
ausencia de evidencia de que funcione. Ningún archivo del plan ni del paper puede decir que falla bajo
las dos versiones mayores.

**Grupo A, incorporación del 2026-08-04.** `Qwen2.5-1.5B-Instruct` entra al grupo A con evidencia
propia (`PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template`, sondeado dentro de la imagen ya
construida `slm-domotica-2026:granite-4-0-350m` con `transformers==4.57.6` verificado en el
contenedor; log en `.claude-scratch/logs/probe_qwen25_15b_groupA.log`). Pin **heredado**
(`motivo_pin == ""`): la sonda prueba membresía de grupo, no necesidad de pin, porque no se corrió su
contraparte bajo 5.14.1 — el mismo estándar de evidencia que las otras filas heredadas del grupo A.
El grupo A activo pasa de 8 a **9** modelos y el grupo B activo de 4 a **3**.

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

### F5 — Espacio de datos `data/2026/` (re-congelado)

**Invariantes re-congelados:**

- `data/2026/detalle_2026.csv` tiene exactamente **384** filas (**12** modelos x 32 comandos), no 448.
- `data/2026/detalle/` tiene exactamente **12** CSV por modelo, uno por modelo del roster activo.
- Los 12 valores distintos de `modelo` coinciden con `roster_activo()`.
- **Invariante nuevo (correspondencia versión↔CSV):** en cada CSV por modelo, la columna
  `transformers_version` es constante y **compatible con el `transformers_pin` de ese modelo**
  (empieza con `4.57.` para el grupo A, con `5.` para el grupo B). Un CSV producido bajo una versión
  que no es la fijada para su modelo es inválido y debe rehacerse.
- `data/2026/detalle/granite-4-0-350m.csv` (producido bajo 4.57.6) **se conserva**: 4.57.6 sigue
  siendo el pin definitivo de `granite-4.0-350m`, así que satisface el invariante y no hay que
  recomputarlo. Condición operativa: el subtask 05 lo rehace con `--force` si el operador no puede
  atestiguar que se produjo en host ocioso bajo `--cpus=2 --memory=8g --cpuset-cpus=0-1`.
- **Invariante nuevo (2026-08-04, validez en todo checkpoint).** El espacio de datos **no gana
  archivos por modelo**: la persistencia incremental de RF20 se logra escribiendo los **mismos**
  archivos de este árbol de forma que cada uno sea **válido y parseable en todo momento**, no
  acumulando parciales con otros nombres. Concretamente: `detalle/<slug>.csv` se escribe completo
  cuando ese modelo termina (nunca a medias); `etiquetas_errores.csv` y `categorias_comandos.csv` se
  reescriben enteros después de cada lote, así que en cualquier instante contienen un prefijo
  consistente con el esquema congelado de más abajo. Un CSV truncado a mitad de línea es un bug, no un
  parcial legítimo (escritura atómica: se escribe a `<archivo>.tmp` y se renombra).
- **Archivo nuevo (2026-08-04):** `data/2026/fallos_barrido.json`. Registra los modelos que fallaron
  de forma irrecuperable durante el barrido, para que el paper pueda documentar la exclusión con su
  error textual (RF20c, F12). Es una **lista** de dicts con exactamente estas claves:
  `modelo`, `hf_repo_id`, `transformers_pin`, `codigo_salida`, `error_textual`, `momento_iso`. Si el
  barrido no tuvo fallos, el archivo es la lista vacía `[]` — **siempre existe** después del subtask
  05, para que no haya ambigüedad entre "no hubo fallos" y "no se registró".

```
data/2026/
  detalle/<slug>.csv          # una por modelo (subtask 05), commiteada al terminar ese modelo
  fallos_barrido.json         # fallos irrecuperables del barrido, [] si no hubo (subtask 05)
  log_barrido.txt             # log de la corrida (subtask 05)
  detalle_2026.csv            # consolidado (subtask 06)
  resumen_etapa1.json         # resumen sin métrica laxa (subtask 06)
  juez_seleccionado.json      # (subtask 06)
  etiquetas_errores.csv       # (subtask 08), reescrito por lote
  categorias_comandos.csv     # (subtask 08), reescrito por lote
  log_juez.txt                # log de la etapa 2 (subtask 08)
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

### F8 — Tablas LaTeX: `src/generate_tex_tables.py` (re-congelado)

Escribe en `paper/02_reescrito/tablas/` cinco fragmentos, cada uno un bloque `\begin{table}...\end{table}` autónomo, sin preámbulo, apto para `\input`:

| archivo | contenido | fuente |
|---|---|---|
| `tabla1_modelos.tex` | roster activo: modelo, params, tier, familia, modo de prompting | `models_2026.roster_activo()` |
| `tabla2_resultados_globales.tex` | JSON válido, exactitud estricta, exactitud laxa, latencia, **estricta publicada (2025)**, **latencia publicada (2025)** | `resumen_2026.json` + `data/resultados_experimento_resumen.json` (F0) |
| `tabla3_por_categoria.tex` | exactitud por categoría lingüística × modelo | `exactitud_por_categoria.csv` |
| `tabla4_taxonomia.tex` | taxonomía de 7 etiquetas × modelo | `taxonomia_2026.csv` |
| `tabla5_versiones.tex` | matriz `modelo \| transformers_pin \| versión resuelta \| ¿necesario? \| motivo` | `models_2026.roster_activo()` |

`tabla5_versiones.tex` debe ser **la misma información** que la matriz de `docker/README.md`. Las demás tablas se generan sobre los **12** modelos del roster activo, nunca sobre las 15 filas del registro.

**Amendado el 2026-08-04 — las dos columnas de continuidad de la Tabla 2 (RF19b).** `tabla2_resultados_globales.tex` gana **exactamente dos** columnas al final, tomadas de `data/resultados_experimento_resumen.json` (archivo **F0**, que se **lee** y jamás se modifica):

- `Estricta 2025 (\%)` ← `exact_match_pct` publicado de ese modelo;
- `Latencia 2025 (s)` ← `avg_latencia_s` publicado de ese modelo.

El *join* es por el campo `modelo`, que coincide textualmente entre los dos JSON para los **tres** modelos presentes en ambos estudios (`SmolLM2-360M-Instruct`, `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`). Para los otros **9** modelos del roster activo, ambas celdas llevan literalmente `--` (dos guiones; en LaTeX se renderiza como raya). El orden final de columnas de la Tabla 2 queda congelado así:

```
Modelo | JSON válido (\%) | Estricta (\%) | Laxa (\%) | Latencia (s) | Estricta 2025 (\%) | Latencia 2025 (s)
```

Se agregan **columnas**, no una tabla nueva: el paper sigue teniendo **cinco** fragmentos `.tex`, y todas las afirmaciones de "cinco tablas" del plan y del TEST_PLAN siguen válidas. Decisión de diseño: una sexta tabla habría obligado a tocar F9, el conteo de `tablas/*.tex` y cuatro checks del TEST_PLAN, con más superficie de inconsistencia y ningún beneficio de presentación — la comparación se lee mejor al lado de la cifra nueva.

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

### F11 — Docker (re-congelado)

```
docker/
  Dockerfile.modelo        # parametrizado por ARG TRANSFORMERS_PIN
  requirements-base.txt    # todo menos transformers
  build_all.py             # construye una imagen por modelo del roster ACTIVO
  run_sweep.py             # corre las 12 imágenes del roster activo, de a una, en orden de registro
  README.md                # matriz de versiones + motivos (RF5) + tabla de exclusiones
```

- `docker/run_sweep.py` y `docker/build_all.py` operan **por defecto** sobre **`roster_activo()`**
  (12), no sobre `MODELOS_2026` (15). Pedir explícitamente por `--modelo` / `--desde` un modelo
  excluido falla con `ValueError` en español que nombra su `motivo_exclusion`. Eso ahora incluye a
  `Qwen3.5-2B`.
- La invocación congelada para modelos *gated* (con `--env-file <repo>/.env`) **se conserva tal
  cual**; simplemente ningún modelo del roster activo la usa. El fail-fast de credenciales también
  se conserva. `Qwen3.5-2B` está excluido pero **no** es gated: su invocación **no** lleva
  `--env-file`.

**Amendado el 2026-08-04 — `--cpuset-cpus` (RNF1, RNF6).** Toda invocación de `docker run` de este
proyecto —las 12 del barrido y la del juez— agrega `--cpuset-cpus=<par de núcleos>` junto a
`--cpus=2 --memory=8g`. El valor congelado por defecto es **`0-1`**, y es **el mismo para las 12
corridas y para el juez**: lo que hace comparable la latencia no es qué par de núcleos se use, sino
que sea **el mismo par en todas las corridas**. `docker/run_sweep.py` expone `--cpuset` (default
`"0-1"`) para poder cambiarlo en otro host, y el valor efectivamente usado queda registrado en
`docker/README.md` junto a las versiones efectivas. Razón por la que el flag es necesario y no
redundante con `--cpus=2`: `--cpus` es una **cuota** del planificador CFS (permite migración entre
núcleos y no reserva nada), mientras que `--cpuset-cpus` **fija** los núcleos y elimina la competencia
por tiempo de CPU y la variabilidad por migración. Lo que **no** puede hacer —y por eso la ejecución
es secuencial— es particionar la caché L3 ni el bus de memoria, que es donde la inferencia de LLM en
CPU está realmente limitada.

Nombre de imagen congelado: `slm-domotica-2026:<slug>`. Invocación congelada por modelo, según `gated`:

Modelos del roster activo (12) — sin credenciales:
```
docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
  -v <repo>/data:/app/data \
  -v <repo>/.hf_cache:/app/.hf_cache \
  slm-domotica-2026:<slug> \
  python src/run_sweep_2026.py --modelo "<nombre>"
```

Modelos **gated** (2, definidos pero excluidos del roster activo) — se agrega **exclusivamente**
`--env-file <repo>/.env`, y solo se usa si alguien los reactiva explícitamente:
```
docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
  --env-file <repo>/.env \
  -v <repo>/data:/app/data \
  -v <repo>/.hf_cache:/app/.hf_cache \
  slm-domotica-2026:<slug> \
  python src/run_sweep_2026.py --modelo "<nombre>"
```

Etapa 2 (juez) — mismos límites y mismo `--cpuset-cpus`, para que el entorno de las dos etapas sea el
mismo:
```
docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
  -v <repo>/data:/app/data \
  -v <repo>/.hf_cache:/app/.hf_cache \
  slm-domotica-2026:<slug del juez> \
  python src/judge_2026.py
```

Un único volumen de caché HF compartido por todas las imágenes; los árboles de dependencias quedan aislados. `docker/README.md` debe contener (i) la matriz de versiones con la columna `¿necesario?` para los 12 del roster activo, (ii) una tabla de **exclusiones** con **las tres** filas —los dos gated con el endpoint que devolvió 403 y la fecha, y `Qwen3.5-2B` con su causa (compatibilidad no verificada bajo 5.14.1, decisión del usuario del 2026-08-04)— y (iii) el valor de `--cpuset-cpus` usado. La matriz debe ser la **misma información** que la Tabla 5 del paper. El texto dice "corre las **12** imágenes del roster activo", nunca 14 ni 15.

**Credenciales (congelado).** El token de HuggingFace vive **solo** en `.env` en la raíz del repo, ignorado por git. Se inyecta **únicamente en tiempo de ejecución** vía `docker run --env-file .env`, y **solo** a las imágenes con `gated = True` si se las reactiva explícitamente; el roster activo corre sin credenciales. Está **prohibido**: declarar `ARG HF_TOKEN` o `ENV HF_TOKEN` en cualquier `Dockerfile`, pasar el token como `--build-arg`, hacer `COPY .env`, escribirlo en una capa de imagen, o invocar `huggingface-cli login` / `huggingface_hub.login()`. La única forma de consumirlo es leer `os.environ["HF_TOKEN"]` en runtime. Los archivos del plan y del repo lo referencian **solo** como `$HF_TOKEN`; su valor no aparece en ningún archivo versionado.

**Fallo temprano (congelado).** Antes de construir o correr nada, si la lista de modelos a ejecutar incluye algún `gated` y `.env` no existe o `HF_TOKEN` está ausente/vacío, se aborta con `ValueError` en español que nombre el modelo y el archivo faltante — **nunca** a mitad de la descarga. El mensaje de error **no** imprime el valor del token. Como el roster activo no tiene modelos gated, este camino solo se ejerce si alguien reactiva un modelo excluido.

### F12 — Persistencia incremental y continuación ante fallo (nuevo, 2026-08-04)

Contrato que gobierna al **orquestador del barrido** (`docker/run_sweep.py`) y al **juez**
(`src/judge_2026.py`). Es la forma ejecutable de **RF20**. Motivo registrado: esta sesión ya perdió
trabajo tres veces (reinicio del backend de Docker, salida del proceso, error 529). Lo que solo existe
en el working tree es exactamente lo que cuesta una caída.

**F12.1 — El barrido no corta al primer fallo.** `docker/run_sweep.py` cambia de "corta al primer
fallo" a "registra y sigue":

```python
FALLOS_PATH: Path                # <repo>/data/2026/fallos_barrido.json
CPUSET_POR_DEFECTO: str = "0-1"

def comando_run(modelo: ModeloEvaluado2026, raiz: Path,
                force: bool = False, cpuset: str = CPUSET_POR_DEFECTO) -> list[str]: ...

def ejecutar_barrido(modelos: list[ModeloEvaluado2026], force: bool, dry_run: bool,
                     cpuset: str = CPUSET_POR_DEFECTO) -> int: ...
```

Semántica congelada de `ejecutar_barrido`:

- Itera **todos** los modelos de la lista, en orden de registro, uno por vez. Un código de salida
  distinto de 0 en un modelo **no** interrumpe la iteración.
- Por cada fallo acumula un dict con exactamente las claves de F5
  (`modelo`, `hf_repo_id`, `transformers_pin`, `codigo_salida`, `error_textual`, `momento_iso`) y lo
  imprime a stderr en el momento, con el error textual completo.
- Al terminar escribe `FALLOS_PATH` con la lista acumulada — **siempre**, incluso vacía (`[]`), salvo
  en `--dry-run`.
- Devuelve `0` si no hubo ningún fallo y `1` si hubo al menos uno, **después** de haber intentado los
  12. Nunca devuelve temprano por un fallo de modelo.
- **AUTH-STOP sigue siendo la excepción:** un 401/403 **sí** corta inmediatamente y no se reintenta
  (ver F11 y el protocolo de fallo del subtask 05). "Continuar ante fallo" aplica a fallos técnicos del
  modelo, **no** a fallos de autenticación.

**F12.2 — Commit por modelo, con la versión efectiva en el mensaje.** Cuando `data/2026/detalle/<slug>.csv`
aparece completo para un modelo, se commitea **de inmediato**, antes de arrancar el siguiente. Plantilla
de mensaje congelada (la versión sale de la columna `transformers_version` de ese mismo CSV, o sea la
que realmente corrió dentro del contenedor, no la pedida):

```
data(2026): barrido de <nombre del modelo> (transformers <version efectiva>)
```

`git add` acotado al CSV de ese modelo (F1: nunca `git add -A`). Prohibido acumular 12 resultados sin
versionar.

**F12.3 — El juez persiste por lote.** `src/judge_2026.py` gana un parámetro de persistencia en sus dos
etiquetadores y dos flags:

```python
Persistidor = Callable[[pd.DataFrame], None]

def etiquetar_errores(df_detalle: pd.DataFrame, juez,
                      persistir: Persistidor | None = None) -> pd.DataFrame: ...
def etiquetar_categorias(df_detalle: pd.DataFrame, juez,
                         persistir: Persistidor | None = None) -> pd.DataFrame: ...
```

- `persistir` se invoca con el **DataFrame acumulado hasta ese punto** (columnas ya en el orden de
  `COLUMNAS_ETIQUETAS` / `COLUMNAS_CATEGORIAS`): en `etiquetar_errores`, al terminar cada **modelo**;
  en `etiquetar_categorias`, al terminar cada **comando**.
- `persistir is None` ⇒ comportamiento idéntico al de antes de este contrato (útil en los tests
  unitarios), así que los tests ya escritos del subtask 07 siguen valiendo sin cambios.
- El `main()` pasa un persistidor que escribe el CSV **atómicamente** (`<archivo>.tmp` + `rename`), de
  modo que el archivo en disco siempre es parseable (F5).
- Flags nuevos: `--reanudar` (opt-in) salta los pares `(modelo, idx)` / los `idx` que ya están en el
  CSV en disco; `--cpuset` no aplica acá (el flag vive en la invocación de `docker run`, F11).
  **`--reanudar` es opt-in a propósito**: el default recalcula todo, así que el check de determinismo
  (AC5 / C12) sigue midiendo determinismo real y no la trivialidad de saltear todo.

**F12.4 — Exclusión por fallo, documentada.** Todo modelo que quede en `fallos_barrido.json` (a) queda
fuera del consolidado y de todas las tablas y figuras, (b) **rompe** el invariante de 384 filas, así que
el subtask 06 debe reportarlo explícitamente en vez de dejar pasar un consolidado corto, y (c) se
documenta en el paper como **limitación**, con la misma redacción prudente que `Qwen3.5-2B`: se dice
qué falló y con qué error, no se especula sobre por qué. Si `fallos_barrido.json` no es `[]`, el conteo
esperado del consolidado pasa a `(12 − |fallos|) × 32` y ese número tiene que quedar escrito en el
commit del subtask 06 y en el paper.

## 5. Cross-task acceptance

Se verifica al final del ciclo (`create-test-plan` / `run-test-plan`), no dentro de ningún subtask:

1. **Regresión legacy intacta.** `pytest -q` verde y `git diff main --stat` no muestra cambios en ninguno de los archivos de F0.
2. **Barrido completo y consistente.** `data/2026/detalle_2026.csv` tiene exactamente **384** filas (**12 modelos x 32** comandos), los **12** valores de `modelo` coinciden con `roster_activo()`, y no hay `idx` faltantes ni duplicados por modelo.
3. **Cobertura de la etapa 2.** Toda fila con `match_exact == False` en `detalle_2026.csv` tiene una fila correspondiente en `etiquetas_errores.csv` con al menos una etiqueta válida, y `categorias_comandos.csv` cubre los 32 `idx` exactamente una vez.
4. **Coherencia estricta/laxa.** Para cada modelo de `resumen_2026.json`, `exact_match_laxo_pct >= exact_match_pct`, y ambos en `[0, 100]`.
5. **Determinismo del juez.** Reejecutar la etapa 2 sobre el mismo `detalle_2026.csv` reproduce `etiquetas_errores.csv` y `categorias_comandos.csv` byte a byte.
6. **Trazabilidad de versiones.** Los 12 modelos del roster activo aparecen en la matriz de `docker/README.md` y en `paper/02_reescrito/tablas/tabla5_versiones.tex` con su pin, su versión resuelta y si el pin es necesario o heredado; los dos grupos de versión y el confusor que introducen se mencionan en `03_metodologia.tex` y en `06_amenazas.tex`.
7. **Los dos papers compilan.** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` termina con código 0 en `paper/01_original/` y en `paper/02_reescrito/`, produciendo `main.pdf` en ambas.
8. **Envío ciego.** `python scripts/check_anonimato.py paper/02_reescrito` sale con código 0, incluyendo la revisión de metadatos del `main.pdf` ya construido.
9. **Números del paper == números de los datos.** Cada valor de las Tablas 2/3/4 del PDF reescrito proviene de los fragmentos generados por `src/generate_tex_tables.py`; regenerar los fragmentos no produce diff.
10. **Amenazas nuevas presentes.** `06_amenazas.tex` cubre explícitamente las **tres** amenazas de RF16.
11. **Historial limpio.** Un commit por subtask en `feat/reescritura-experimento-2026`, y `paper_cacic_LNCS_word.docx` sigue sin trackear, y `.env` sigue sin trackear (`git ls-files` no lista `.env` ni ningún `*.env`).
12. **Ningún secreto versionado.** `git grep -iE 'hf_[A-Za-z0-9]{20,}'` sale vacío (exit 1) en todo el árbol trackeado — el patrón es estricto a propósito: un token de HuggingFace es `hf_` + ~34 alfanuméricos, mientras que `hf_repo_id`, `.hf_cache`, `HF_TOKEN` y `HF_HOME` son identificadores legítimos del diseño y deben seguir existiendo. Además, `git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env' docker/` sale vacío, y `git ls-files` no lista `.env`.
13. **Exclusiones declaradas.** Los modelos excluidos (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`, `Qwen3.5-2B`, y `Qwen2.5-0.5B-Instruct` desde el Delta 05) figuran en el registro con `activo=False` y `motivo_exclusion` no vacío, y en la tabla de exclusiones de `docker/README.md`, con sus causas distintas correctamente atribuidas (acceso no otorgado para los dos primeros, compatibilidad **no verificada** para `Qwen3.5-2B`, baseline-completion para `Qwen2.5-0.5B-Instruct`); ninguno figura en ninguna tabla ni figura de resultados. **Actualizado (Delta 04, post-`8fa6380`):** la redacción original de este ítem exigía además "limitación explícita en `03_metodologia.tex` y `06_amenazas.tex`" — esa prosa de exclusiones se retiró deliberadamente del paper (la selección de los doce se presenta en positivo, por sus criterios de inclusión); `docker/README.md` es, desde entonces, la única fuente textual obligatoria de la documentación de exclusiones, y este ítem ya no exige duplicarla en el paper. Ver Check C14 de `TEST_PLAN.md`.
14. **Correspondencia versión↔CSV.** Para cada CSV de `data/2026/detalle/`, la columna `transformers_version` es constante y compatible con el `transformers_pin` del modelo correspondiente (grupo A → `4.57.*`, grupo B → `5.*`). Sobre el roster activo eso son **9** CSV con `4.57.*` y **3** con `5.*`.
15. **Generación anterior medida ahora, sin columnas de continuidad (RF19, retractado por el Delta 04).** No hay un "paper 2025": fue un borrador de este mismo trabajo que nunca se publicó ni se publicará, y sus cifras se produjeron con el entorno sin fijar (`transformers`, `torch`, `attn_implementation` y `revision` sin pin), por lo que no son comparables de forma válida. `paper/02_reescrito/tablas/tabla2_resultados_globales.tex` tiene **cinco** columnas (Modelo, JSON válido, Estricta, Laxa, Latencia) y ninguna cifra de esa ronda anterior; `Qwen2.5-1.5B-Instruct` sigue en el roster activo, pero como representante de la generación anterior de SLMs medido bajo el mismo protocolo 2026 que el resto — no por continuidad con datos de 2025. `04_resultados.tex` reporta esa comparación generacional (\texttt{granite-4.0-1b} 90,6\% vs. \texttt{Qwen2.5-1.5B-Instruct} 65,6\%) y `05_discusion.tex` la interpreta. Este ítem reemplaza al RF19 original (columnas de continuidad contra `data/resultados_experimento_resumen.json`); ver Check C15 de `TEST_PLAN.md`.
16. **Persistencia incremental verificable (RF20, F12).** El historial de la rama tiene **un commit por modelo** del barrido, cada uno con el nombre del modelo y su versión efectiva de `transformers` en el asunto (`git log --oneline main..HEAD --grep='^data(2026): barrido de'` lista tantos commits como CSV haya en `data/2026/detalle/`), y `data/2026/fallos_barrido.json` existe —`[]` si no hubo fallos—. Cada CSV de `data/2026/detalle/` es parseable por pandas sin error (ningún parcial truncado).
17. **Envelope de recursos completo.** Las **12** invocaciones planeadas por `docker/run_sweep.py --dry-run` llevan `--memory=8g`, `--cpus=2` y `--cpuset-cpus` con **un único** valor idéntico en las doce; ninguna menciona GPU; y ese valor está registrado en `docker/README.md`.

## 6. Tareas globales de test

Insertadas por `/create-test-plan`. **No son nodos del DAG de §3**: son tareas a nivel índice que cubren dos verificaciones *cross-task* que ningún bloque `Verify` de subtask puede cubrir, porque cruzan artefactos de subtasks distintos. Se ejecutan con el resto del ciclo y las consume `TEST_PLAN.md` (Test 2 y Test 3). Cada una lleva su propio commit, igual que un subtask (RNF5).

### G1 — `tests/test_credenciales_gated.py`: fallo temprano por credencial ausente

**depends_on:** [01, 04, 16, 17] — cruza el flag `gated` de `src/models_2026.py` (subtask 01) con `validar_credenciales` de `docker/run_sweep.py` (subtask 04) y con `roster_activo()` (subtask 16). Depende también de **17** porque el conteo de no-gated del registro pasa de 12 a **13** con el intercambio de roster (`Qwen3.5-2B` queda excluido pero no es gated). Ninguno de los tres lo testeaba antes del Delta 02: el subtask 04 testea que `--env-file` se agregue solo a los gated, pero **no** que el barrido aborte cuando falta el token, y `roster_activo()` no existía hasta el subtask 16. Es el invariante de cierre del *Delta 01* y el que evita que un barrido largo muera a mitad de camino. **Se conserva como test válido del mecanismo** tras el Delta 02: ningún modelo del roster activo necesita `$HF_TOKEN` (cero gated en los 12), así que se agrega la comprobación de que `roster_activo()` no tiene gated y de que `validar_credenciales(roster_activo(), raiz)` devuelve `None` **sin** `.env`; el aborto sigue testeado pasando explícitamente `gated()`.

**Archivos:** `tests/test_credenciales_gated.py`

- [ ] Escribir `tests/test_credenciales_gated.py` completo:

  ```python
  """Invariante del Delta 01 / F11: si la corrida incluye modelos gated y falta
  el token, el barrido aborta ANTES de construir o descargar nada, con un
  mensaje que jamás imprime el valor del token."""

  import importlib.util
  import sys
  from pathlib import Path

  import pytest

  RAIZ = Path(__file__).resolve().parent.parent
  sys.path.insert(0, str(RAIZ / "src"))

  from models_2026 import MODELOS_2026, gated, por_nombre, roster_activo  # noqa: E402

  # Valor ficticio deliberadamente sin la forma `hf_` + 20+ alfanuméricos, para
  # que el escaneo de secretos (AC12) no lo tome por un token real.
  TOKEN_FICTICIO = "valor-de-prueba-no-es-un-token"


  def _cargar_run_sweep():
      """`docker/` no es un paquete importable: se carga el módulo por ruta."""
      ruta = RAIZ / "docker" / "run_sweep.py"
      spec = importlib.util.spec_from_file_location("run_sweep_docker", ruta)
      modulo = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(modulo)
      return modulo


  @pytest.fixture(scope="module")
  def run_sweep():
      return _cargar_run_sweep()


  def test_aborta_si_no_existe_el_archivo_env(run_sweep, tmp_path):
      with pytest.raises(ValueError) as exc:
          run_sweep.validar_credenciales(gated(), tmp_path)
      assert ".env" in str(exc.value)


  def test_aborta_si_el_token_esta_vacio(run_sweep, tmp_path):
      (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
      with pytest.raises(ValueError) as exc:
          run_sweep.validar_credenciales(gated(), tmp_path)
      assert "HF_TOKEN" in str(exc.value)


  def test_aborta_si_el_token_no_esta_declarado(run_sweep, tmp_path):
      (tmp_path / ".env").write_text("OTRA_COSA=1\n", encoding="utf-8")
      with pytest.raises(ValueError):
          run_sweep.validar_credenciales(gated(), tmp_path)


  @pytest.mark.parametrize("nombre", ["gemma-3-270m-it", "Llama-3.2-1B-Instruct"])
  def test_un_solo_modelo_gated_ya_dispara_el_aborto(run_sweep, tmp_path, nombre):
      with pytest.raises(ValueError):
          run_sweep.validar_credenciales([por_nombre(nombre)], tmp_path)


  def test_los_no_gated_corren_sin_credenciales(run_sweep, tmp_path):
      # 13 tras el Delta 2026-08-04: los 12 del roster activo + Qwen3.5-2B, que
      # esta excluido pero NO es gated. El subtask 17 lo renombra y ajusta el
      # conteo; el commit b6b54a1 no se reescribe.
      no_gated = [m for m in MODELOS_2026 if not m.gated]
      assert len(no_gated) == 13
      assert run_sweep.validar_credenciales(no_gated, tmp_path) is None


  def test_el_roster_activo_no_tiene_ningun_modelo_gated():
      assert len(roster_activo()) == 12
      assert all(not m.gated for m in roster_activo())


  def test_el_roster_activo_corre_sin_credenciales_aunque_falte_env(run_sweep, tmp_path):
      # tmp_path no tiene .env: si roster_activo() necesitara alguna credencial, esto abortaria.
      assert run_sweep.validar_credenciales(roster_activo(), tmp_path) is None


  def test_no_aborta_con_el_token_presente(run_sweep, tmp_path):
      (tmp_path / ".env").write_text(f"HF_TOKEN={TOKEN_FICTICIO}\n", encoding="utf-8")
      assert run_sweep.validar_credenciales(list(MODELOS_2026), tmp_path) is None


  def test_leer_el_token_no_lo_imprime(run_sweep, tmp_path, capsys):
      ruta_env = tmp_path / ".env"
      ruta_env.write_text(f"HF_TOKEN={TOKEN_FICTICIO}\n", encoding="utf-8")
      assert run_sweep._hf_token_de_env(ruta_env) == TOKEN_FICTICIO
      capturado = capsys.readouterr()
      assert TOKEN_FICTICIO not in capturado.out
      assert TOKEN_FICTICIO not in capturado.err


  def test_el_mensaje_de_aborto_no_filtra_el_token(run_sweep, tmp_path):
      (tmp_path / ".env").write_text(f"HF_TOKEN=   \nOTRA={TOKEN_FICTICIO}\n", encoding="utf-8")
      with pytest.raises(ValueError) as exc:
          run_sweep.validar_credenciales(gated(), tmp_path)
      assert TOKEN_FICTICIO not in str(exc.value)
  ```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_credenciales_gated.py`
- [ ] Confirmar que el valor ficticio no dispara el escaneo de secretos:
  `git grep -iE 'hf_[A-Za-z0-9]{20,}' -- tests/test_credenciales_gated.py` → vacío (exit 1)
- [ ] Correr la suite completa: `pytest -q`
- [ ] `git add tests/test_credenciales_gated.py`
- [ ] `git commit -m "test: fallo temprano del barrido si falta \$HF_TOKEN para los modelos gated"`

> **Nota post-Delta 02 (2026-08-03).** El commit de arriba ya está hecho (`b6b54a1`). Las dos
> pruebas de `roster_activo()` agregadas más arriba en este bloque documentan la extensión
> pendiente: `roster_activo` no existe hasta que el subtask **16** lo agrega a `models_2026.py`.
> Aplicar ese código real a `tests/test_credenciales_gated.py` es trabajo del subtask 16
> (ver `TODO_16_roster-12-y-pins.md`, criterio de aceptación de `pytest -q`), no de esta fusión de
> documentación. No se reescribe el commit `b6b54a1`.
>
> **Nota post-Delta 03 (2026-08-04).** El subtask 16 ya aplicó ese código (`864c694`), pero con
> `assert len(no_gated) == 12`. Con el intercambio de roster el registro pasa a **15** modelos y a
> **13** no-gated (`Qwen3.5-2B` queda excluido y no es gated), así que ese test **falla** hasta que el
> subtask **17** lo ajuste. Es exactamente la clase de fallo que se quiere: el conteo está escrito y
> el test lo defiende. El bloque de arriba ya refleja el estado corregido (test renombrado a
> `test_los_no_gated_corren_sin_credenciales`, conteo 13); aplicarlo a
> `tests/test_credenciales_gated.py` es trabajo del subtask 17. No se reescribe `864c694`.

### G2 — `tests/test_integracion_2026.py`: integridad de los artefactos del pipeline

**depends_on:** [01, 05, 06, 08, 09, 16, 17] — verifica de una sola vez los criterios AC2, AC3, AC4 y la correspondencia versión↔CSV de F5, que abarcan artefactos producidos por subtasks distintos y por eso no son atribuibles a ninguno. Depende también de **16** porque el conteo pasa de 448 (14×32) a **384 (12×32)** y porque la nueva correspondencia `transformers_version`↔`transformers_pin` necesita el registro con `activo`/`motivo_pin` ya corregido, y de **17** porque el reparto de esos 12 CSV entre grupos pasa a **9** con `4.57.*` y **3** con `5.*`. El bloque de código de abajo ya es agnóstico del reparto: deriva todo de `roster_activo()` y del `transformers_pin` de cada fila, así que no hay conteos que ajustar a mano. Los tests están guardados por `skipif` sobre la existencia de los artefactos, de modo que `pytest -q` sigue verde en cada commit desde el primero (RNF4) y el archivo se puede crear apenas termine el subtask 01.

**Archivos:** `tests/test_integracion_2026.py`

- [ ] Escribir `tests/test_integracion_2026.py` completo:

  ```python
  """AC2/AC3/AC4: integridad cross-task de `data/2026/`. Solo lee artefactos ya
  producidos; nunca reejecuta el barrido ni al juez."""

  import json
  import sys
  from pathlib import Path

  import pandas as pd
  import pytest

  RAIZ = Path(__file__).resolve().parent.parent
  sys.path.insert(0, str(RAIZ / "src"))

  from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo  # noqa: E402
  from taxonomia_2026 import (  # noqa: E402
      CATEGORIAS_LINGUISTICAS,
      ETIQUETAS_ERROR,
      SEP_ETIQUETAS,
  )

  DIR_2026 = RAIZ / "data" / "2026"
  DETALLE = DIR_2026 / "detalle_2026.csv"
  ETIQUETAS = DIR_2026 / "etiquetas_errores.csv"
  CATEGORIAS = DIR_2026 / "categorias_comandos.csv"
  RESUMEN = DIR_2026 / "resumen_2026.json"
  DETALLE_DIR = DIR_2026 / "detalle"

  N_COMANDOS = 32
  N_FILAS = len(roster_activo()) * N_COMANDOS  # 384 (12 x 32)

  requiere_barrido = pytest.mark.skipif(
      not DETALLE.exists(), reason=f"falta {DETALLE.name}: requiere el subtask 06"
  )
  requiere_etapa2 = pytest.mark.skipif(
      not (ETIQUETAS.exists() and CATEGORIAS.exists()),
      reason="faltan los CSV de la etapa 2: requiere el subtask 08",
  )
  requiere_resumen = pytest.mark.skipif(
      not RESUMEN.exists(), reason=f"falta {RESUMEN.name}: requiere el subtask 09"
  )


  @requiere_barrido
  def test_el_consolidado_tiene_384_filas_del_roster_activo():
      df = pd.read_csv(DETALLE)
      assert len(df) == N_FILAS
      assert sorted(df["modelo"].unique()) == sorted(m.nombre for m in roster_activo())
      for nombre, grupo in df.groupby("modelo"):
          assert sorted(grupo["idx"]) == list(range(N_COMANDOS)), f"{nombre}: idx incompletos o duplicados"


  @requiere_barrido
  def test_la_version_de_transformers_de_cada_csv_corresponde_a_su_pin():
      """F5: la columna transformers_version de cada CSV por modelo es constante y
      compatible con el transformers_pin de ese modelo (grupo A -> 4.57.*, grupo B -> 5.*)."""
      for modelo in roster_activo():
          ruta = DETALLE_DIR / f"{modelo.nombre.lower().replace('.', '-').replace(' ', '-')}.csv"
          if not ruta.exists():
              pytest.skip(f"falta {ruta.name}: requiere el subtask 05")
          df = pd.read_csv(ruta)
          versiones = df["transformers_version"].unique()
          assert len(versiones) == 1, f"{modelo.nombre}: transformers_version no es constante"
          version = versiones[0]
          if modelo.transformers_pin == BASELINE_TRANSFORMERS:
              assert str(version).startswith("4.57."), f"{modelo.nombre}: esperaba 4.57.*, salio {version}"
          elif modelo.transformers_pin == TRANSFORMERS_5X:
              assert str(version).startswith("5."), f"{modelo.nombre}: esperaba 5.*, salio {version}"
          else:
              pytest.fail(f"{modelo.nombre}: transformers_pin fuera de los dos grupos conocidos")


  @requiere_barrido
  @requiere_etapa2
  def test_la_etapa_2_cubre_exactamente_las_respuestas_incorrectas():
      df = pd.read_csv(DETALLE)
      etq = pd.read_csv(ETIQUETAS)
      incorrectas = {
          (f.modelo, f.idx) for f in df[~df["match_exact"].astype(bool)].itertuples()
      }
      etiquetadas = {(f.modelo, f.idx) for f in etq.itertuples()}
      assert incorrectas == etiquetadas
      for texto in etq["etiquetas"]:
          partes = [p for p in str(texto).split(SEP_ETIQUETAS) if p]
          assert partes, "hay una fila etiquetada con el conjunto vacío"
          assert set(partes) <= set(ETIQUETAS_ERROR), f"etiqueta fuera de vocabulario: {texto!r}"


  @requiere_etapa2
  def test_las_categorias_cubren_los_32_comandos_una_sola_vez():
      cat = pd.read_csv(CATEGORIAS)
      assert sorted(cat["idx"]) == list(range(N_COMANDOS))
      assert set(cat["categoria"]) <= set(CATEGORIAS_LINGUISTICAS)


  @requiere_resumen
  def test_la_exactitud_laxa_domina_a_la_estricta_en_los_12_modelos():
      resumen = json.loads(RESUMEN.read_text(encoding="utf-8"))
      assert [f["modelo"] for f in resumen] == [m.nombre for m in roster_activo()]
      for fila in resumen:
          estricta, laxa = fila["exact_match_pct"], fila["exact_match_laxo_pct"]
          assert 0 <= estricta <= 100, f"{fila['modelo']}: estricta fuera de rango"
          assert 0 <= laxa <= 100, f"{fila['modelo']}: laxa fuera de rango"
          assert laxa >= estricta, f"{fila['modelo']}: laxa {laxa} < estricta {estricta}"
  ```

- [ ] Correr y confirmar que **saltea** (todavía no hay artefactos): `pytest -q -rs tests/test_integracion_2026.py` → `5 skipped`, con el motivo nombrando el subtask que falta.
- [ ] Correr la suite completa: `pytest -q`
- [ ] `git add tests/test_integracion_2026.py`
- [ ] `git commit -m "test: integridad cross-task de data/2026 (448 filas, cobertura de etapa 2, laxa >= estricta)"`

> **Nota post-Delta 02 (2026-08-03).** El commit de arriba ya está hecho (`fd45589`), con el
> conteo de 448 filas vigente en ese momento. El bloque de código de este documento ya refleja
> el estado corregido (384 filas, `roster_activo()`, test de correspondencia versión↔pin); aplicar
> ese código real a `tests/test_integracion_2026.py` es trabajo de una fase de implementación
> posterior (depende de `roster_activo()`, que introduce el subtask **16**), no de esta fusión de
> documentación. No se reescribe el commit `fd45589`.

- [ ] **Después del subtask 09**, reejecutar y confirmar que ya **no** saltea:
  `pytest -q -rs tests/test_integracion_2026.py` → `5 passed`, cero `skipped`.

# Delta 04 — corrida de control con el prompt original (subtask 18, BLOQUEANTE)

Decisión del usuario/root del 2026-08-04, tomada tras cerrar el barrido de los 12 modelos
(`fallos_barrido.json == []`, 384 filas en `data/2026/detalle/`): los tres anclajes de continuidad
con la Tabla 2 publicada (RF19) divergen del barrido 2026 con **signo opuesto** según el modelo:

| anclaje | estricta publicada | estricta nueva (prompt 2026) | delta | json publicado → nuevo |
|---|---|---|---|---|
| `SmolLM2-360M-Instruct` | 18.8% | 9.4% | **−9.4 pp** | 100% → **25%** |
| `Qwen2.5-1.5B-Instruct` | 50.0% | 65.6% | **+15.6 pp** | 100% → 100% |
| `SmolLM2-1.7B-Instruct` | 59.4% | 50.0% | **−9.4 pp** | 100% → 100% |

Como el signo varía por modelo, **ninguna** comparación nuevo-vs-publicado puede aislar el efecto
del prompt endurecido (RF1) del efecto del modelo: el sesgo no se puede argumentar siquiera en una
única dirección con los datos que había hasta el 2026-08-04. Se decide una corrida de **control**
que aísla esa variable.

**Invariante de diseño, y es todo el punto:** la corrida de control difiere del barrido 2026 en
**exactamente una variable** — el prompt de sistema. Mismo dataset, mismos 32 comandos en el mismo
orden, misma decodificación (greedy, temperatura 0), mismo código de scoring, mismas imágenes
Docker, mismo envolvente de recursos. Solo `SYSTEM_PROMPT_PAPER` (el prompt original del paper,
importado tal cual de `src/prompt.py`, F0 congelado) reemplaza a `SYSTEM_PROMPT_2026`.

**Qué se implementó (código y tests; la corrida real la lanza el orquestador después de este
reporte):**

- `src/run_control_prompt_original.py` (nuevo): entrypoint que corre dentro del contenedor. Reusa
  `run_sweep_2026.evaluar_modelo` tal cual — no lo copia — vía un `ConstructorEntrada` inyectable
  (factorización mínima y aditiva de `run_sweep_2026.py` y `prompt_2026.py`, con default idéntico al
  comportamiento actual del barrido principal). Restringido a los tres anclajes
  (`ANCLAS_CONTROL`); escribe `data/2026/control_prompt_original/<slug>.csv` con el mismo esquema
  `COLUMNAS_DETALLE` y el mismo invariante de 32 filas/`idx` que el barrido principal.
- `docker/run_control.py` (nuevo): runner host-side, mirroring `docker/run_sweep.py` y sus dos fixes
  ya probados (`subprocess.run(..., encoding="utf-8", errors="replace")`; continuación-ante-fallo con
  `data/2026/control_prompt_original/fallos_control.json`, que siempre existe salvo `--dry-run`).
  Secuencial, un anclaje a la vez, mismo `--memory=8g --cpus=2 --cpuset-cpus=0-1`. **No reconstruye
  ninguna imagen**: reusa `slm-domotica-2026:{smollm2-360m-instruct,smollm2-1-7b-instruct,
  qwen2-5-1-5b-instruct}` tal cual del barrido principal (los tres anclajes son grupo A).
- `src/comparar_control.py` (nuevo): genera
  `data/2026/control_prompt_original/comparacion_tres_bandas.json` con las tres bandas (publicado,
  control, nuevo) y sus tres deltas de exactitud estricta y de tasa de JSON válido, por anclaje. Cifras
  publicadas leídas de `data/resultados_experimento_resumen.json` (F0), nunca hardcodeadas. La
  latencia promedio se reporta con una nota explícita: no es comparable entre bandas si la máquina no
  estuvo ociosa durante la corrida de control.
- `tests/test_control_prompt_original.py` (nuevo, TDD): cubre las tres piezas de arriba con dobles de
  tokenizer y filesystem aislado (`tmp_path`/`monkeypatch`), sin inferencia real y sin escribir bajo
  `data/2026/`.

**Por qué es BLOQUEANTE para el paper.** La corrida de control (`docker/run_control.py`, sin
`--dry-run`) tiene que ejecutarse y su resultado tiene que leerse **antes** de escribir la Sección 4
del paper reescrito: si la banda de control reproduce la cifra publicada (dentro de un margen
razonable), la divergencia nuevo-vs-publicado es atribuible al endurecimiento del prompt y así se
redacta. **Si la banda de control NO reproduce la cifra publicada, hay un defecto del harness que hay
que encontrar y corregir antes de redactar cualquier interpretación de RF19** — escribir la Sección 4
sobre datos de un harness no validado sería presentar como hallazgo lo que podría ser un bug. Por eso
el subtask **18** (`depends_on: [05]`) se agrega al DAG y los subtasks **14** y **15** (paper
reescrito, secciones 1–3 y 4–7) pasan a depender también de **18**, además de sus dependencias
previas: `14.depends_on` pasa de `[10, 11, 12, 13]` a `[10, 11, 12, 13, 18]`, y `15.depends_on` pasa
de `[14]` a `[14, 18]`. Ningún texto de resultados o discusión sobre los tres anclajes puede
escribirse antes de que `18` esté cerrado con la corrida real ejecutada.

**Qué NO cambia.** El barrido principal (`data/2026/detalle/`, 12 CSV, `fallos_barrido.json == []`)
es de solo lectura para todo este delta: no se reinterpreta, no se reejecuta, no se toca. La corrida
de control es un experimento adicional y aislado, no una corrección del barrido.

# Delta 05 — la corrida de control no reprodujo la Tabla 2 publicada; baseline re-medido (subtask 19)

Decisión del usuario/root del 2026-08-04, tomada tras leer el resultado real de la corrida de
control del subtask 18 (`docker/run_control.py`, sin `--dry-run`, ejecutada por el orquestador):
**NO reprodujo** la Tabla 2 publicada para los tres anclajes de continuidad.

| anclaje | publicado | control (prompt original) | nuevo (prompt 2026) |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | 50.0% | **56.2%** | 65.6% |
| `SmolLM2-1.7B-Instruct` | 59.4% | **50.0%** | 50.0% |
| `SmolLM2-360M-Instruct` | 18.8% | **0.0%** | 9.4% |

Como el invariante de diseño del subtask 18 fijaba **exactamente una** variable distinta entre la
corrida de control y el barrido principal (el prompt de sistema, `SYSTEM_PROMPT_PAPER` en vez de
`SYSTEM_PROMPT_2026`) y el resto —dataset, decodificación greedy/temperatura 0, código de scoring,
imágenes Docker, envolvente de recursos— se mantuvo idéntico, esta divergencia demuestra que **el
prompt no fue la causa principal** de la brecha nuevo-vs-publicado que motivó el Delta 04: si lo
fuera, la banda de control tendría que acercarse a la publicada, y en cambio la banda de control
diverge de la publicada casi tanto como la banda nueva (y con signo distinto según el anclaje).

**Causa, con evidencia (no la versión de `transformers` en sí, que es irrecuperable — no perseguirla):**
el `requirements.txt` legacy del repo fija `transformers>=4.46.0` **sin cota superior**, y
`data/resultados_experimento_detalle.csv` (F0, congelado) **no tiene columna `transformers_version`**.
El entorno de la corrida original nunca quedó fijado en el momento de correrla ni registrado en sus
resultados: no hay ningún dato en el repo del que se pueda recuperar qué versión de `transformers` (ni
qué revisión exacta de los pesos de HuggingFace, silenciosamente actualizables) produjo la Tabla 2
publicada. **Es irreproducibilidad del trabajo original, no un defecto del harness 2026.**

**Acotación explícita — dónde NO está el problema.** `tests/test_metricas.py` sigue **verde** y
recalcula los porcentajes publicados de la Tabla 2 a partir de los outputs **crudos** publicados
(`data/resultados_experimento_detalle.csv`, F0): la ruta de *scoring* es fiel, byte a byte, a la
lógica legacy. La divergencia está enteramente en la **generación** (versión de librería, revisiones
de pesos actualizadas en silencio, configuración de decodificación no registrada), nunca en cómo se
puntúan las respuestas. **No auditar ni modificar la ruta de scoring: no es ahí.**

## Decisión: baseline re-medido e internamente consistente

Las cifras publicadas dejan de ser un baseline **validado** y pasan a **referencia histórica, no
reproducida** (así queda marcado explícitamente en `src/comparar_baseline.py:NOTA_PUBLICADO` y en el
JSON que produce). Las comparaciones del paper usan en cambio un baseline **RE-MEDIDO** e
internamente consistente: los **4** modelos del paper original, medidos bajo el mismo harness/prompt
2026, misma máquina, temperatura 0 — mismas condiciones que el barrido de los 12, así que la
comparación es de igual a igual. Tres de los 4 (`SmolLM2-360M-Instruct` 9.4%, `SmolLM2-1.7B-Instruct`
50.0%, `Qwen2.5-1.5B-Instruct` 65.6%) ya existían en `data/2026/detalle/` (son las anclas de
continuidad RF19, `activo=True`). El cuarto, `Qwen2.5-0.5B-Instruct` (publicado 43.8%), faltaba —
subtask 19 lo agrega.

**Qué implementó el subtask 19 (código y tests listos; la corrida real de
`docker/run_baseline.py`, sin `--dry-run`, la lanza el orquestador después de este reporte):**

1. **Sonda de grupo de versión, no inferida.** `Qwen2.5-0.5B-Instruct` se sondeó (no se asumió por
   familia) dentro de la imagen ya construida del grupo A (`slm-domotica-2026:granite-4-0-350m`,
   `transformers==4.57.6`): `PROBE_OK|Qwen2.5-0.5B-Instruct|4.57.6|chat_template`. Grupo A, pin
   heredado (no se probó 5.14.1), mismo estándar de evidencia que `Qwen2.5-1.5B-Instruct`.
2. **`src/models_2026.py`**: campo nuevo `baseline_original: bool` (default `False`, ortogonal a
   `activo`) y `roster_baseline_original()` (los 4 modelos). Fila nueva de
   `Qwen2.5-0.5B-Instruct` con `activo=False` + `motivo_exclusion` que aclara que es
   baseline-completion (no un modelo nuevo del roster 2026, ni un bloqueo de acceso/versión — una
   tercera causa de exclusión distinta de las dos ya existentes). Registro **15 → 16**; roster
   activo se mantiene en **12** (decisión del usuario, no revisitada).
3. **`docker/build_all.py`**: `seleccionar_modelos()` deja de rechazar un modelo `activo=False`
   cuando además es `baseline_original=True`. Imagen construida por ese mismo path, sin build
   ad-hoc: `slm-domotica-2026:qwen2-5-0-5b-instruct`. No se reconstruyó ninguna otra imagen.
4. **`src/run_baseline_original.py`** (nuevo) + **`docker/run_baseline.py`** (nuevo): mismo patrón
   que el subtask 18 (`run_control_prompt_original.py` / `run_control.py`), pero sin sustituir el
   prompt — reusa `run_sweep_2026.evaluar_modelo` con su prompt 2026 por defecto — y restringido a
   los `baseline_original` que **todavía no están activos** (hoy, solo `Qwen2.5-0.5B-Instruct`).
   Escribe `data/2026/baseline_original/<slug>.csv` con el mismo `COLUMNAS_DETALLE` y el mismo
   invariante de 32 filas/`idx`. Mismo envolvente de recursos (`--memory=8g --cpus=2
   --cpuset-cpus=0-1`), mismos dos fixes ya probados (`encoding="utf-8", errors="replace"`;
   continuación ante fallo con `data/2026/baseline_original/fallos_baseline.json`, siempre existe
   salvo `--dry-run`), y el mismo montaje de `src/` de solo lectura que ya costó una corrida fallida
   por su ausencia en el subtask 18.
5. **`src/comparar_baseline.py`** (nuevo): publicado (histórico) vs re-medido para los 4 modelos,
   con su delta; determina el **mejor original re-medido** y la **comparación central** (mejor
   arquitectura 2026 vs mejor original re-medido) **por código**, leyendo el CSV real de
   `granite-4.0-1b` en vez de afirmar un número a mano. Escribe
   `data/2026/baseline_original/comparacion_baseline.json`.

**Hallazgo a reportar, no a asumir.** Con los 3 CSV reales que ya existen (falta solo
`Qwen2.5-0.5B-Instruct`), el **mejor original re-medido HOY es `Qwen2.5-1.5B-Instruct` al 65.6%**,
no `SmolLM2-1.7B-Instruct` al 50.0%: `Qwen2.5-1.5B-Instruct` sube fuerte bajo el prompt 2026 (50.0%
publicado → 65.6% re-medido). La comparación central resulta, con los datos de hoy, `granite-4.0-1b`
(90.6%) vs `Qwen2.5-1.5B-Instruct` (65.6%) — delta **25.0 pp**, no 40.6 pp. Esto se recalcula
automáticamente en cuanto corra el cuarto modelo (`python src/comparar_baseline.py` de nuevo): si
`Qwen2.5-0.5B-Instruct` no supera 65.6%, el ganador no cambia; si lo hiciera, el script lo reflejaría
sin que nadie tenga que tocar el número a mano.

## Bookkeeping del DAG

Subtask **19** (`depends_on: [05, 18]`) se agrega al DAG. `09.depends_on` pasa de `[08]` a `[08,
19]` y `10`/`11` heredan la dependencia por transitividad (mismo patrón que `17`). `14.depends_on`
pasa de `[10, 11, 12, 13, 18]` a `[10, 11, 12, 13, 18, 19]` y `15.depends_on` de `[14, 18]` a `[14,
18, 19]`.

**Los subtasks 09, 10, 11, 14 y 15 deben usar el baseline RE-MEDIDO, no las cifras publicadas.**
Ningún texto ni tabla de esos cinco subtasks puede presentar la Tabla 2 publicada como si fuera un
baseline validado contra el que comparar los 12 modelos del roster 2026: la comparación válida es
contra `data/2026/baseline_original/comparacion_baseline.json` (o, equivalentemente, contra las
métricas re-medidas de `roster_baseline_original()`). Donde la cifra publicada se mencione, debe ir
acompañada de la nota de referencia histórica no reproducida (`comparar_baseline.NOTA_PUBLICADO`).

**La irreproducibilidad es un aporte secundario para §5/§6, no una nota de Amenazas a la Validez.**
El hallazgo de que la Tabla 2 publicada no es reproducible bajo un control con el prompt idéntico —y
la evidencia de por qué (versión de dependencia sin cota superior, sin registro de entorno)— es en sí
mismo un resultado metodológico del paper reescrito, no un matiz a enterrar en la sección de
limitaciones. Debe discutirse en Resultados/Discusión (§5/§6) con la misma prioridad que la
comparación de arquitecturas 2026, precisamente porque motiva por qué el paper usa un baseline
re-medido en vez del publicado para su comparación central.

**Qué NO cambia.** `data/2026/detalle/*.csv`, `data/2026/control_prompt_original/*.csv` y los F0
congelados son de solo lectura para este delta: no se reinterpretan, no se reejecutan, no se tocan.
La corrida de baseline-completion es un experimento adicional y aislado (un modelo, una vez), no una
corrección del barrido principal ni de la corrida de control.

# Delta 06 — corrección de framing: borrador (no publicado), auditoría retirada, titular con prompt fijo

Decisión del usuario/root del 2026-08-04, que corrige hacia adelante los Deltas 04 y 05 **sin
editarlos**: `paper_cacic_LNCS_word.docx` es un **BORRADOR** del paper que se está escribiendo, no
trabajo publicado. Los Deltas 04/05 lo trataban como si fuera la Tabla 2 publicada de un paper ya
existente ("publicado", "referencia histórica no reproducida", "auditoría de reproducibilidad");
esa premisa era incorrecta y sus consecuencias, todas vinculantes, son estas:

1. Las cifras del borrador (`SmolLM2-360M-Instruct` 18.8, `Qwen2.5-0.5B-Instruct` 43.8,
   `Qwen2.5-1.5B-Instruct` 50.0, `SmolLM2-1.7B-Instruct` 59.4) son valores de **borrador SUPERADOS**,
   no una base de comparación. No hay "publicado vs re-medido" que reportar.
2. **La irreproducibilidad-como-aporte queda RETIRADA.** El Delta 05 afirmaba ("La irreproducibilidad
   es un aporte secundario para §5/§6, no una nota de Amenazas a la Validez") que no reproducir la
   Tabla 2 publicada era en sí mismo un resultado metodológico a discutir en Resultados/Discusión. Eso
   presuponía que había algo publicado que "fallar en reproducir" — no lo hay: es un borrador propio,
   todavía sin someter. No se puede publicar "no reprodujimos nuestro propio borrador" como
   contribución. Ese texto del Delta 05 no se edita (forward-only); este delta lo anula hacia adelante.
3. Los 4 modelos de `roster_baseline_original()` (`SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`,
   `Qwen2.5-1.5B-Instruct`, `Qwen2.5-0.5B-Instruct`) son el **brazo de generación ANTERIOR** de este
   mismo experimento, no "anclas de continuidad con la literatura": se miden bajo el mismo harness,
   prompt, máquina, versiones fijadas y decodificación greedy que los modelos más nuevos (Granite 4 /
   Qwen3.5 / LFM2.5 / OLMo-2). Eso hace que la comparación **generacional** sea internamente válida
   sin apoyarse en ningún número del borrador.
4. El control (`control_prompt_original/`) y la baseline-completion (`baseline_original/`) sobreviven
   solo como justificación **metodológica** de por qué se re-midió en vez de reusar las cifras del
   borrador — un párrafo de método, nunca una sección de resultados.

## El titular se calcula con el prompt fijo, nunca mezclando bandas

Un error real ya cometido: comparar una celda medida bajo el prompt 2026 (`granite-4.0-1b`, 90.6%)
contra una celda del brazo anterior medida bajo el prompt ORIGINAL producía un delta de ~40 pp que no
correspondía a nada real — el prompt no era la misma variable a ambos lados. La comparación central
del paper es, en cambio, **mejor arquitectura 2026 vs mejor generación anterior, ambas bajo el mismo
prompt 2026**:

| modelo | prompt original | prompt 2026 |
|---|---|---|
| `SmolLM2-360M-Instruct` | 0.0% | 9.4% |
| `SmolLM2-1.7B-Instruct` | 50.0% | 50.0% |
| `Qwen2.5-1.5B-Instruct` | 56.2% | **65.6%** |
| `Qwen2.5-0.5B-Instruct` | pendiente | 21.9% |

Bajo el prompt 2026, el mejor de los 4 es `Qwen2.5-1.5B-Instruct` (65.6%). Titular:
`granite-4.0-1b` (90.6%) vs `Qwen2.5-1.5B-Instruct` (65.6%) — delta **25.0 pp, no 40 pp**.
`src/comparar_baseline.py:comparar_celdas()` es una guardia explícita que **RECHAZA**
(`raise ValueError`) comparar celdas de distinto prompt, para que ese error no se pueda repetir en
silencio.

## Qué implementó este delta (código y tests, en verde)

1. **`src/run_control_prompt_original.py` / `docker/run_control.py`**: la validación del entrypoint
   de control se amplía de los tres anclajes hardcodeados a los 4 modelos de
   `roster_baseline_original()`; `docker/run_control.py` suma `--modelo` para correr solo el modelo
   faltante (`Qwen2.5-0.5B-Instruct`) sin tocar ni recomputar los tres CSV de control ya existentes.
   Ninguna imagen Docker se reconstruye.
2. **`src/comparar_baseline.py`**: reemplaza la comparación "publicado vs re-medido" por la matriz
   4×2 real (4 modelos × {prompt original, prompt 2026}), con exact-match y json_válido por celda, el
   efecto de prompt por modelo y el titular calculado por código con el prompt fijo. Sin lenguaje de
   auditoría. El artefacto se renombra de `data/2026/baseline_original/comparacion_baseline.json`
   (borrado) a `data/2026/matriz_generacion_anterior.json`.
3. **`driver_control_05b.sh`** (scratchpad, no versionado): corre
   `docker/run_control.py --modelo "Qwen2.5-0.5B-Instruct"`, la única celda pendiente de la matriz.

## Limitación conocida para §6 (no se persigue)

`attn_implementation` y la `revision` de cada modelo de HuggingFace quedaron **sin pinnear** en todo
el barrido 2026 (roster activo, control y baseline-completion): cada corrida usó lo que
`transformers` resolviera por defecto en el momento de ejecutarse, y ningún CSV registra qué
implementación de atención ni qué revisión exacta de pesos se usó. Pinnearlos ahora exigiría
re-correr el barrido completo (32 comandos × 12 modelos activos, más control y baseline-completion)
para que la comparación siga siendo de igual a igual — no vale la pena: se documenta como limitación
explícita en §6, no se corrige.

## Qué NO cambia

`data/2026/detalle/*.csv`, `data/2026/control_prompt_original/*.csv` (los tres ya existentes) y
`data/2026/baseline_original/qwen2-5-0-5b-instruct.csv` son de solo lectura para este delta: no se
reinterpretan, no se reejecutan, no se tocan. La única corrida real pendiente es la que completa la
matriz (`Qwen2.5-0.5B-Instruct` bajo el prompt original), que lanza el orquestador después de este
reporte.
