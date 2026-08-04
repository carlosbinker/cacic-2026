# Test plan — Reejecución del experimento con roster 2026, verificación automática en dos etapas y reescritura del paper en LaTeX

**Source TODO:** `TODO.md`
**Goal:** Probar que el DAG completo (subtasks 01–17 + G1/G2, tras el Delta 2026-08-04) produce un barrido de 12 modelos del roster activo coherente de punta a punta, dos papers LaTeX que compilan y pasan el filtro de envío ciego, y un árbol versionado sin secretos ni regresiones sobre el material congelado de F0.

**Scope:** Solo checks **globales / cross-task**: los criterios de §5 *Cross-task acceptance* del índice (extendido a **17** puntos por el Delta 2026-08-04), más los invariantes de los tres deltas: roster activo de **12** con cero gated sobre un registro de **15**; **3** excluidos con **dos causas distintas** (2 por acceso de descarga no otorgado, 1 por compatibilidad de versión **no verificada**); dos grupos de versión de `transformers` mutuamente excluyentes, repartidos **9 / 3** sobre el roster activo; plomería de `$HF_TOKEN` conservada pero no exigida por el roster activo; envelope de recursos con **pinning de núcleos** idéntico en las 12 corridas; **persistencia incremental** con un commit por modelo y **continuación ante fallo**; **continuidad con la Tabla 2 publicada** sobre los 3 modelos presentes en ambos estudios; y cero secretos versionados. Fuera de alcance acá: todo lo que sea atribuible a un único subtask — vive en el bloque `Verify` de su `TODO_<NN>_<slug>.md` y **no** se reproduce en este archivo. También fuera de alcance lo que el índice declara *out of scope* en §2.4 (login interactivo, `Qwen2.5-0.5B-Instruct`, sondear `Qwen3.5-2B` bajo 5.14.1, ejecución concurrente de varios modelos, abstract en inglés, chequeo del límite de 10 páginas, GPU/cuantización, juez externo por API). **Ojo:** `Qwen2.5-1.5B-Instruct` **sí** está en alcance — el Delta 2026-08-04 lo reincorporó al roster activo.

Nota deliberada sobre solape: los criterios AC1–AC14 están declarados por el índice como *"se verifica al final del ciclo (`create-test-plan` / `run-test-plan`), no dentro de ningún subtask"*. Algunos subtasks hacen un *smoke check* del mismo hecho en el momento de su propio commit (p. ej. 15 compila los papers). Este plan los reejecuta como **compuerta de regresión sobre el estado final del árbol**, que es un hecho distinto del smoke check puntual: un fix posterior puede romperlos.

**Estructura de costo — leer antes de correr nada.** El tier caro de este proyecto **no es una API paga**: es el barrido de los 12 modelos del roster activo en Docker, CPU-only, `--memory=8g --cpus=2 --cpuset-cpus=0-1`, **estrictamente secuencial** (~7 h de reloj, ver RNF6), más las descargas, y la reejecución del juez. La distinción que gobierna todo el plan es:

- **Producir** `data/2026/**` es caro (E1, E2 de §6).
- **Verificar** `data/2026/**` es barato: son lecturas de CSV/JSON con pandas, segundos. Los checks AC2/AC3/AC4 son `cheap` aunque dependan de artefactos que costó 9 h generar.

Por lo tanto `/run-test-plan --scope=cheap` corre todo salvo E1, E2 y C12, y es seguro en cada iteración de fix.

---

## 1. Pre-flight

Ejecutar desde la raíz del repo. **Usar la shell Bash (Git Bash), no PowerShell**: todos los comandos de este plan usan sintaxis POSIX (`wc -l`, heredocs `<<'PY'`, `&&`, exit codes).

**P1 — Rama y árbol de trabajo**
Run: `git rev-parse --abbrev-ref HEAD && git status --porcelain | head -20`
Expected: imprime `feat/reescritura-experimento-2026`. El `git status` puede listar untracked esperados (`paper_cacic_LNCS_word.docx`, `.env`, `LaTeX2e (1)/`, `data/2026/`, `.hf_cache/`) pero **ningún** archivo modificado sin commitear.

**P2 — Dependencias de Python**
Run: `python -c "import pandas, matplotlib, pytest, pypdf; print('deps OK')"`
Expected: imprime `deps OK`. Si falla, correr `pip install -r requirements.txt` y reintentar. (`pypdf` lo agrega el subtask 12 para `revisar_pdf`.)

**P3 — Toolchain de LaTeX**
Run: `latexmk -version`
Expected: exit 0 e imprime la versión de latexmk. Si no está, los checks C7/C8 y el build de §6 no pueden correr.

**P4 — Compuerta del tier caro (solo antes de E1/E2/C12; omitir en `--scope=cheap`)**
Run: `docker info --format '{{.ServerVersion}}' && echo "docker: OK"`
Expected: imprime la versión del server de Docker y `docker: OK`. Tras el Delta 02, el roster activo (12) tiene **cero** modelos *gated*, así que E1/E2 **no** requieren `.env` ni `HF_TOKEN`. Esa plomería sigue existiendo (T2 la sigue verificando como mecanismo) pero solo se ejerce si el barrido incluye explícitamente uno de los **dos** modelos excluidos **por acceso** — en ese caso sí hace falta `.env` con `HF_TOKEN` no vacío antes de arrancar. El tercer excluido, `Qwen3.5-2B`, no necesita credenciales (no es *gated*).

Dos condiciones más antes de E1/E2, agregadas por el Delta 2026-08-04:

```bash
# El almacenamiento del daemon esta en modo escritura (el 2026-08-03 quedo en solo lectura
# a mitad de ronda de sondeo y ningun build ni run pudo correr).
docker run --rm slm-domotica-2026:granite-4-0-350m python -c "print('escritura y run OK')"
```

Expected: imprime `escritura y run OK`. Si reaparece `read-only file system`, **parar**: reparar Docker Desktop es una decisión del usuario, que está usando la máquina, y no se intenta desde el plan.

Y **host ocioso** (RNF1, RNF6): nada más consumiendo CPU mientras corre E2, ni siquiera otro check del tier `cheap` que use Docker. La latencia es el resultado que esta condición protege, y contaminarla es invisible en la tabla final.

---

## 2. Automated tests

### Test 1 — Suite completa de pytest
**Covers AC:** AC1 (*Regresión legacy intacta*), y de forma agregada los invariantes unitarios de F3/F4/F11 escritos por los subtasks.
**Cost:** `cheap`
**Run:** `pytest -q`
**Expected:** exit 0. La línea final reporta solo `passed` (y eventualmente `skipped`, ver Test 3): **cero** `failed` y **cero** `error`. Incluye `tests/test_metricas.py` sin modificar (F0).
**On failure indicates:** una regresión de contrato — un módulo nuevo rompió `scoring.py`/`schema.py`, o un subtask editó un archivo congelado de F0, o un invariante de `models_2026`/`taxonomia_2026` dejó de valer.

### Test 2 — Fallo temprano por credencial ausente (invariante del Delta 01)
**Covers AC:** RF17 / F11 *"Fallo temprano (congelado)"* y el cierre del Delta 01. Es un check **cross-task**: cruza `src/models_2026.py` (subtask 01, flag `gated`) con `docker/run_sweep.py` (subtask 04, `validar_credenciales`). Ningún subtask lo testea hoy — lo crea la tarea global **G1**.
**Cost:** `cheap`
**Run:** `pytest -q tests/test_credenciales_gated.py`
**Expected:** exit 0, todos los tests `passed`. En particular: con una raíz temporal sin `.env`, `validar_credenciales(gated(), tmp)` levanta `ValueError`; con `HF_TOKEN` vacío también; con la lista de los **13** no-gated del registro (12 activos + `Qwen3.5-2B`, excluido pero **no** gated) o con `roster_activo()` (12) **no** levanta nada aunque falte `.env`; y `_hf_token_de_env` no escribe nada en stdout. El conteo de no-gated pasó de 12 a **13** con el subtask 17: antes de que 17 cierre, este test **falla** con `assert 13 == 12`, y eso es lo esperado — el arreglo es cerrar 17, no relajar el conteo.
**On failure indicates:** un barrido que incluyera un modelo *gated* reactivado podría abortar a mitad de la descarga en vez de fallar en el segundo 0 — exactamente el modo de fallo caro que el Delta 01 vino a cerrar. O peor: el mensaje de error filtra el token.

### Test 3 — Integridad de los artefactos del barrido y de la etapa 2
**Covers AC:** AC2 (*384 filas, 12 modelos del roster activo, sin `idx` faltantes ni duplicados*), AC3 (*cobertura de la etapa 2*), AC4 (*coherencia estricta/laxa*). Lo crea la tarea global **G2**.
**Cost:** `cheap` — son lecturas de `data/2026/*.csv` y `resumen_2026.json` con pandas. **No** reejecuta el barrido.
**Run:** `pytest -q -rs tests/test_integracion_2026.py`
**Expected:** exit 0. Antes de que el subtask 09 haya corrido, se aceptan `skipped` con motivo explícito (los tests están guardados por `skipif` sobre la existencia de los artefactos). En la **pasada confirmatoria** (post-subtask 09): `4 passed`, **cero** `skipped` — un `skipped` ahí significa que falta un artefacto que debería existir.
**On failure indicates:** el consolidado perdió o duplicó filas (bug en `stage1_2026.py`), el juez no cubrió todas las respuestas incorrectas o etiquetó fuera del vocabulario cerrado (bug en `judge_2026.py` o en el fallback de parseo de F5), o la métrica laxa quedó por debajo de la estricta (bug en `metrics_2026.py`, viola la definición congelada de F6).

---

## 3. Manual checks

### Check 1 — Legibilidad de las figuras con 12 modelos
**Covers AC:** RF12 (*barras horizontales agrupadas por tier, nombres largos legibles, sin apiñamiento a 12 modelos*). El subtask 10 verifica por script que los PNG existen, pesan >20 KB y contienen 12 modelos separados 6/6 por tier; **no** puede verificar que se lean bien. Eso es irreductiblemente humano.
**Cost:** `cheap`
**Steps:**
1. Abrir `figures/2026/fig1_exactitud_latencia_2026.png` con un visor de imágenes al 100 % de zoom.
2. Leer las 12 etiquetas del eje Y de arriba a abajo.
3. Abrir `figures/2026/fig2_exactitud_por_campo_2026.png` y repetir.
**Expected observation:** los 12 nombres del roster activo (`LFM2.5-230M` … `Qwen2.5-1.5B-Instruct`, en orden de registro) se leen completos, sin truncado con `…` ni superposición entre etiquetas vecinas; el bloque `sub-1B` (6 barras) está visualmente separado del bloque `1-2B` (6 barras); en el panel izquierdo de fig1 se distinguen dos series superpuestas (estricta y laxa) con leyenda; **ninguno de los tres modelos excluidos** (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`, `Qwen3.5-2B`) aparece, y **sí** aparece `Qwen2.5-1.5B-Instruct`, que es el nombre más largo del roster y por lo tanto el peor caso de legibilidad del eje Y.
**On failure indicates:** el layout de `generate_figures_2026.py` no escaló de 8 a 12 modelos, o un modelo excluido se coló en el gráfico — la figura no es publicable aunque el script salga con exit 0. Si falta `Qwen2.5-1.5B-Instruct`, la figura se generó contra el roster previo al subtask 17 y hay que regenerarla.

### Check 2 — El PDF reescrito renderiza tablas y figuras
**Covers AC:** AC7 (*los dos papers compilan*) y AC9 (*números del paper == números de los datos*) en su dimensión visual: `latexmk` puede salir con 0 y aun así producir tablas desbordadas o `??` en las referencias cruzadas.
**Cost:** `cheap`
**Steps:**
1. Abrir `paper/02_reescrito/main.pdf`.
2. Localizar las cinco tablas (`tab:modelos`, `tab:globales`, `tab:categorias`, `tab:taxonomia`, `tab:versiones`) y las dos figuras.
3. Buscar en el texto la cadena `??`.
**Expected observation:** las **cinco** tablas y las dos figuras aparecen, ninguna se sale del margen de la caja de texto LNCS, la tabla 1 lista 12 filas de modelos (el roster activo, sin los **3** excluidos), y no hay ninguna ocurrencia de `??` (referencia cruzada rota) en todo el PDF. **Prestar atención especial a la Tabla 2**, que pasó de 5 a **7** columnas con las dos de continuidad 2025 (F8): es la más ancha del paper y la primera candidata a desbordar la caja LNCS. Sus dos últimas columnas deben mostrar valores solo en **3** de las 12 filas y `--` en las otras 9.
**On failure indicates:** un fragmento `.tex` generado por `generate_tex_tables.py` es demasiado ancho para el formato LNCS —muy probablemente la Tabla 2 de 7 columnas—, o un `\label`/`\ref` quedó desparejado entre secciones. Si la Tabla 2 desborda, el arreglo es acortar los encabezados o el `spec` de columnas en `generate_tex_tables.py`, **nunca** quitar las columnas de continuidad: son el mecanismo de RF19 y de C15. Siguen siendo cinco tablas: no se agregó una sexta.

---

## 4. Command checks

Orden recomendado: **C9 antes de C7**, y **C7 antes de C8** (C8 inspecciona metadatos del `main.pdf` que produce C7).

### Check C1 — Inmutabilidad de F0 en el árbol de trabajo
**Covers AC:** AC1 (*`git diff main --stat` no muestra cambios en ninguno de los archivos de F0*).
**Cost:** `cheap`
**Run:**
```bash
git diff --stat main -- \
  data/resultados_experimento_detalle.csv data/resultados_experimento_resumen.json \
  data/dataset_comandos_domotica.csv tests/test_metricas.py \
  src/models.py src/prompt.py src/scoring.py src/schema.py src/metrics.py \
  src/run_evaluation.py src/generate_figures.py src/validar_contra_resultados_originales.py \
  figures/fig1_exactitud_latencia.png figures/fig2_exactitud_por_campo.png
```
**Expected:** salida **vacía** (ni una línea).
**On failure indicates:** un subtask editó material congelado — el corolario de F0 (*todo el código nuevo va en módulos nuevos*) se violó y los resultados publicados dejaron de ser reproducibles.

### Check C2 — Inmutabilidad de F0 en todo el historial de la rama
**Covers AC:** AC1 + AC11 (*historial limpio*). Más fuerte que C1: C1 mira el estado final, C2 mira que **ningún commit intermedio** haya tocado F0.
**Cost:** `cheap`
**Run:**
```bash
git log --oneline main..HEAD -- \
  data/resultados_experimento_detalle.csv data/resultados_experimento_resumen.json \
  data/dataset_comandos_domotica.csv tests/test_metricas.py \
  src/models.py src/prompt.py src/scoring.py src/schema.py src/metrics.py \
  src/run_evaluation.py src/generate_figures.py src/validar_contra_resultados_originales.py \
  figures/fig1_exactitud_latencia.png figures/fig2_exactitud_por_campo.png
git rev-list --count main..HEAD
```
**Expected:** el `git log` da salida **vacía**; el `git rev-list --count` da **≥ 19** (los **17** subtasks del DAG + G1 + G2 de §6 del índice, un commit cada uno como mínimo).
**On failure indicates:** si el `git log` no está vacío, un subtask tocó F0 y otro lo revirtió — el árbol final miente sobre lo que pasó. Si el conteo es < 19, algún subtask no dejó su propio commit y se violó RNF5 (*un commit por subtask*), lo que rompe la trazabilidad que pide AC11.
**Nota sobre el umbral (Delta 2026-08-04).** Es una **cota inferior**, no un conteo exacto, y el número real queda bastante por encima: subtasks de alcance amplio (04, 16) aportaron varios commits, y el subtask **05** aporta ahora **un commit por modelo** (RF20a / F12.2, hasta 12 más su commit de cierre) y el **08** uno por lote. Antes del Delta 2026-08-04 el umbral era 18 (16 nodos + G1 + G2); el nodo 17 lo sube a 19. Nunca convertir esto en una igualdad.

### Check C3 — Ningún token de HuggingFace versionado (árbol completo)
**Covers AC:** AC12 (*Ningún secreto versionado*) y punto 4 del Delta 01.
**Cost:** `cheap`
**Run:** `git grep -iE 'hf_[A-Za-z0-9]{20,}' ; test $? -eq 1 && echo "sin token versionado OK"`
**Expected:** imprime `sin token versionado OK` y nada más. `git grep` sale con 1 (sin coincidencias).
**Nota sobre el patrón — no relajarlo.** El patrón exige `hf_` + **20 o más** alfanuméricos seguidos porque un token real es `hf_` + ~34 alfanuméricos. La forma laxa `hf_[A-Za-z]` es **incorrecta**: matchea identificadores legítimos del diseño (`hf_repo_id` de F3, `.hf_cache` de F11, `HF_TOKEN`, `HF_HOME`) que **deben** seguir existiendo en el árbol. Si este check empieza a dar falsos positivos, el bug está en el patrón, no en el repo.
**On failure indicates:** el valor del token se filtró a un archivo trackeado — probablemente un log commiteado del barrido (`data/2026/log_barrido.txt`), el `docker/README.md`, o un `TODO_<NN>_*.md`. Es un incidente de seguridad: hay que rotar el token, no solo borrar la línea.

### Check C4 — El token nunca entra en una capa de imagen
**Covers AC:** AC12 (segunda mitad) y F11 (*prohibido `ARG`/`ENV HF_TOKEN`, `COPY .env`, `--build-arg`, login interactivo*). El subtask 04 testea esto sobre `docker/`; acá se extiende a **las rutas que realmente pueden terminar en una capa de imagen** — Dockerfiles en cualquier ubicación (hay uno en la raíz, legacy) y el código que esos Dockerfiles copian (`docker/`, `src/`, `scripts/`).
**Cost:** `cheap`
**Run:**
```bash
# Alcance acotado a rutas que pueden terminar en una capa de imagen: Dockerfiles
# en cualquier lado + el código que copian. NO se incluye *.md ni tests/: ambos
# necesariamente CITAN los patrones prohibidos para documentarlos/testearlos,
# y ese texto nunca entra en una capa de imagen. Ampliar el alcance de vuelta
# al árbol completo hace que el check se autofalle contra su propia documentación.
git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env|--build-arg[= ]*HF_TOKEN' -- '*Dockerfile*' 'docker/**' 'src/**' 'scripts/**' ; test $? -eq 1 && echo "sin credenciales en capas OK"
git grep -niE 'huggingface-cli login|huggingface_hub\.login\(' -- '*Dockerfile*' 'docker/**' 'src/**' 'scripts/**' ; test $? -eq 1 && echo "sin login interactivo OK"
```
**Expected:** imprime `sin credenciales en capas OK` y `sin login interactivo OK`, sin ninguna línea de coincidencia.
**On failure indicates:** el token quedaría horneado en una capa de imagen (recuperable por cualquiera que tenga la imagen) o el pipeline dejó de ser no interactivo — ambas cosas están explícitamente prohibidas por F11 y por §2.4.
**Cobertura complementaria (no cheap, ya cubierta por pytest):** este check manual solo mira el estado actual del árbol; la garantía de que **ningún** `Dockerfile` trackeado (en cualquier ubicación) declara/copia el token, hoy y en cualquier commit futuro, la asegura `tests/test_docker_matriz.py::test_ningun_dockerfile_trackeado_tiene_credenciales_ni_login`, que itera `git ls-files '*Dockerfile*'` — no solo `docker/`.

### Check C5 — Credenciales y `.docx` sin trackear
**Covers AC:** AC11 (*`.env` sigue sin trackear, `git ls-files` no lista `.env` ni ningún `*.env`; `paper_cacic_LNCS_word.docx` sigue sin trackear*).
**Cost:** `cheap`
**Run:**
```bash
git ls-files | grep -iE '\.env$|\.docx$' ; test $? -eq 1 && echo "credenciales y docx sin trackear OK"
git check-ignore -v .env paper_cacic_LNCS_word.docx
```
**Expected:** el primer comando imprime `credenciales y docx sin trackear OK` sin listar archivos. El segundo imprime dos líneas, una por archivo, nombrando la regla de `.gitignore` que lo cubre (`.gitignore:<n>:.env:.env` y la del `.docx`).
**On failure indicates:** el token o el fuente del paper entraron al índice de git. Si es `.env`, es un incidente: rotar el token.

### Check C6 — el barrido corre sin `--env-file`; pedir un excluido falla con el motivo
**Covers AC:** RF4 / RF17 / F11 tras los Deltas 02, 2026-08-04 y 05 (*el plan por defecto son 12 invocaciones sobre el roster activo de un registro de 16, ninguna con `--env-file`; pedir explícitamente cualquiera de los **4** modelos excluidos falla con su `motivo_exclusion`; la receta con `--env-file` sigue existiendo y se aplica solo a `gated()` cuando se la fuerza*), y RNF1 (*`--memory=8g --cpus=2 --cpuset-cpus` en las 12 invocaciones*).
**Nota sobre el conteo (Delta 05).** El registro subió de 15 a 16 porque `Qwen2.5-0.5B-Instruct` se agregó para completar el brazo de generación anterior (`baseline_original=True`, `activo=False`): es una exclusión nueva, por una causa nueva (no es gated, no es compatibilidad no verificada), que no cambia el roster activo de 12. El umbral de excluidos pasa de 3 a 4 por esto, no por una regresión.
**Cost:** `cheap` — `--dry-run` solo imprime los comandos; no arranca ningún contenedor ni descarga nada.
**Run:**
```bash
# Secuencia base (alimenta la fila 4 de la tabla de regresion junto con NF1):
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import MODELOS_2026; print(len(MODELOS_2026))"
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import gated; print(len(gated()))"
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import roster_activo; print(len(roster_activo()))"
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import MODELOS_2026; print(len([m for m in MODELOS_2026 if not m.activo]))"
python docker/run_sweep.py --dry-run | grep -c '^==='
python docker/run_sweep.py --dry-run | grep -c -- '--env-file'
```
**Expected:** `16` (registro completo), `2` (gated en el registro), `12` (roster activo), `4` (excluidos), `12` (invocaciones planeadas), `0` (ninguna con `--env-file`).
**On failure indicates:** si el conteo de `--env-file` es mayor que 0, algo reactivó un modelo gated sin que el índice lo sepa (el token se expondría a un contenedor sin necesitarlo). Si los conteos de registro/gated/activo/excluidos no son 16/2/12/4, el registro o el roster activo se desalinearon del contrato de F3 — en particular, menos de `4` excluidos significa que alguna de las cuatro exclusiones documentadas (los dos gated, `Qwen3.5-2B` por compatibilidad no verificada, o `Qwen2.5-0.5B-Instruct` por completar el baseline del Delta 05) se perdió, y `12` roster activo es el invariante que nunca debe moverse.

**Verificaciones adicionales (mismo check, no entran en la fila 4 de la tabla de regresión):**
```bash
# Pedir explicitamente cualquiera de los 4 excluidos falla con el motivo
for m in "gemma-3-270m-it" "Llama-3.2-1B-Instruct" "Qwen3.5-2B" "Qwen2.5-0.5B-Instruct"; do
  python docker/run_sweep.py --desde "$m" --dry-run >/dev/null 2>&1 ; echo "$m exit=$?"
done

# La receta con --env-file sigue definida y se aplica solo si se fuerza sobre un gated
python - <<'PY'
import sys
sys.path.insert(0, "docker"); sys.path.insert(0, "src")
from models_2026 import gated
from run_sweep import comando_run
from pathlib import Path
for m in gated():
    cmd = comando_run(m, Path("."))
    assert "--env-file" in cmd, f"{m.nombre}: la receta con --env-file debe seguir existiendo"
print("receta --env-file conservada para los 2 gated OK")
PY
```
**Expected (adicional):** `exit=1` para los **cuatro** excluidos, cada uno con un mensaje que nombra su `motivo_exclusion` (403/acceso no otorgado para los dos gated; compatibilidad no verificada para `Qwen3.5-2B`; completar el baseline del Delta 05 para `Qwen2.5-0.5B-Instruct`); `receta --env-file conservada para los 2 gated OK`.
**On failure indicates (adicional):** si pedir un excluido no falla, el barrido podría intentar descargar un modelo sin acceso —o correr `Qwen3.5-2B`, cuya compatibilidad no está verificada— y romper a mitad de camino. Si la receta con `--env-file` desapareció, se perdió la plomería de credenciales que el Delta 01 introdujo — necesaria si algún modelo se reactiva en el futuro. Ojo con el caso mixto: `Qwen3.5-2B` está excluido y **no** es gated, así que tiene que fallar por `motivo_exclusion` **sin** que aparezca `--env-file` en ninguna parte.

### Check C7 — Trazabilidad de versiones (dos grupos, roster activo)
**Covers AC:** AC6 tras el Delta 02 (*los 12 modelos del roster activo aparecen en la matriz de `docker/README.md` **y** en `tabla5_versiones.tex` con su pin, su versión resuelta y si es necesario o heredado, **y** los dos grupos se mencionan en `03_metodologia.tex` y en `06_amenazas.tex`*), RF5.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from models_2026 import roster_activo

destinos = {
    "docker/README.md": Path("docker/README.md"),
    "tabla5_versiones.tex": Path("paper/02_reescrito/tablas/tabla5_versiones.tex"),
    "03_metodologia.tex": Path("paper/02_reescrito/secciones/03_metodologia.tex"),
    "06_amenazas.tex": Path("paper/02_reescrito/secciones/06_amenazas.tex"),
}
textos = {k: v.read_text(encoding="utf-8") for k, v in destinos.items()}

fallas = [
    f"{m.nombre} no aparece en {k}"
    for m in roster_activo()
    for k in ("docker/README.md", "tabla5_versiones.tex")
    if m.nombre not in textos[k]
]
fallas += [
    f"{m.nombre}: motivo_pin no aparece en docker/README.md"
    for m in roster_activo()
    if m.motivo_pin and m.motivo_pin[:30] not in textos["docker/README.md"]
]
for k in ("03_metodologia.tex", "06_amenazas.tex"):
    bajo = textos[k].lower()
    if not (("4.57" in bajo) and ("5.14" in bajo or "5.x" in bajo or "transformers>=5" in bajo)):
        fallas.append(f"{k}: no menciona los dos grupos de version")
assert not fallas, fallas
print(f"trazabilidad OK para los {len(roster_activo())} modelos del roster activo, 2 grupos")
PY
```
**Expected:** imprime `trazabilidad OK para los 12 modelos del roster activo, 2 grupos`. Exit 0.
**On failure indicates:** un modelo del roster activo o su motivo quedó sin documentar en la matriz, o el paper no declara los dos grupos de versión. Es un confusor directo de la tabla de latencia y una de las tres amenazas de RF16 — si no está escrita, el paper afirma una comparación que no puede sostener.
**Nota (Delta 2026-08-04).** El bloque de arriba es agnóstico del reparto por grupo porque itera `roster_activo()` y usa el `motivo_pin` de cada fila: sirve igual con el reparto **9 / 3** vigente. Dos consecuencias que sí cambiaron y hay que tener presentes al leer un fallo: (i) `Qwen2.5-1.5B-Instruct` es un modelo **nuevo** del roster activo, así que si falta en `docker/README.md` o en `tabla5_versiones.tex` este check lo va a marcar y el arreglo es documentarlo, no relajar el check; (ii) `Qwen3.5-2B` **ya no** está en `roster_activo()`, así que este check dejó de exigirlo — su presencia en el árbol la cubren C14 (tabla de exclusiones) y el subtask 17.

### Check C8 — Regeneración idempotente de los fragmentos `.tex`
**Covers AC:** AC9 (*cada valor de las Tablas 2/3/4 proviene de los fragmentos generados por `src/generate_tex_tables.py`; regenerar los fragmentos no produce diff*).
**Cost:** `cheap`
**Run:**
```bash
python src/generate_tex_tables.py >/dev/null
git diff --exit-code -- paper/02_reescrito/tablas/ && echo "regeneracion idempotente OK"
```
**Expected:** imprime `regeneracion idempotente OK`; `git diff --exit-code` sale con 0 (sin diferencias).
**On failure indicates:** los números commiteados en el paper no son los que salen de `resumen_2026.json` / `taxonomia_2026.csv` / `exactitud_por_categoria.csv` hoy — alguien editó una tabla a mano, o los datos cambiaron después de generar los fragmentos. En un paper, esto es un error de integridad de resultados.

### Check C9 — Los dos papers compilan
**Covers AC:** AC7 (*`latexmk` termina con código 0 en `paper/01_original/` y en `paper/02_reescrito/`, produciendo `main.pdf` en ambas*).
**Cost:** `cheap` — build local de LaTeX, determinista y offline; ~1–2 min los dos juntos.
**Run:**
```bash
for d in paper/01_original paper/02_reescrito; do
  (cd "$d" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1) \
    && test -f "$d/main.pdf" && echo "$d compila OK" || { echo "$d FALLA"; exit 1; }
done
```
**Expected:** dos líneas: `paper/01_original compila OK` y `paper/02_reescrito compila OK`, exit 0, y ambos `main.pdf` existen.
**On failure indicates:** un fragmento `.tex` generado tiene LaTeX inválido (falló el escapador `_escapar` de F8 sobre `& % $ # _ { } ~ ^ \`), falta un `\input`, o el árbol no es autocontenido (`llncs.cls` / `splncs03.bst` no se copiaron desde `LaTeX2e (1)/`). Correr con `-interaction=nonstopmode` sin `>/dev/null` para leer el error real.

### Check C10 — Envío ciego
**Covers AC:** AC8 (*`python scripts/check_anonimato.py paper/02_reescrito` sale con código 0, incluyendo la revisión de metadatos del `main.pdf` ya construido*), RF15.
**Cost:** `cheap`
**Run:** `python scripts/check_anonimato.py paper/02_reescrito ; echo "exit=$?"`
**Expected:** no imprime ningún `Hallazgo` y termina en `exit=0`. **Debe correrse después de C9**, para que `main.pdf` exista y sus metadatos entren en la revisión.
**On failure indicates:** hay datos identificatorios en el árbol de envío — autores, afiliación, agradecimientos, financiamiento, ORCID, URL del repositorio, o metadatos `/Author` `/Creator` en el PDF. Es motivo de rechazo administrativo en CACIC; no es un detalle cosmético.

### Check C11 — Las tres amenazas nuevas de RF16 están escritas, y la de raw completion no
**Covers AC:** AC10 (*`06_amenazas.tex` cubre explícitamente las tres amenazas de RF16*), RF18.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
import re
from pathlib import Path
texto = Path("paper/02_reescrito/secciones/06_amenazas.tex").read_text(encoding="utf-8").lower()
# EXACTAMENTE TRES amenazas nuevas, no cuatro: el Delta 2026-08-04 no agrego una
# cuarta, sino que amplio la (2) y la (3).
amenazas = {
    "auto-favorecimiento del juez": (("juez",), ("favorec", "sesgo")),
    # (2) ahora tiene que mencionar tambien la exclusion por compatibilidad no
    # verificada, como evidencia de que el confusor de versiones no es teorico.
    "versiones de libreria como confusor de latencia": (("versi",), ("latencia",)),
    # (3) ahora cubre DOS causas de exclusion: acceso no otorgado y compatibilidad
    # no verificada, 3 de 15 en total.
    # "metodológica" lleva tilde en el paper (ortografia correcta); se acepta
    # tambien la forma sin tilde para no volver a quebrar el check si alguien
    # escribe la variante ASCII en el futuro.
    "exclusion por causas ajenas al metodo": (("acceso restringido", "gated"), ("metodologica", "metodológica")),
}
faltan = [
    nombre
    for nombre, (anclas, extras) in amenazas.items()
    if not any(a in texto for a in anclas) or not any(e in texto for e in extras)
]
if "no verificada" not in texto:
    faltan.append("la exclusion por compatibilidad de version no verificada (Qwen3.5-2B)")
assert not faltan, f"amenazas ausentes o incompletas: {faltan}"
# Verificacion NEGATIVA (Delta 02): no puede quedar ninguna afirmacion de que un
# modelo del roster se prompteo por raw completion; esa amenaza fue ELIMINADA.
prohibido = re.search(r"(lfm2\.5-230m|lfm2\.5-350m|modelos base)[^.]{0,80}raw completion", texto)
assert not prohibido, "sobrevive la amenaza eliminada de incomparabilidad por raw completion"
# Verificacion NEGATIVA (Delta 2026-08-04): no se puede sobreafirmar sobre Qwen3.5-2B.
# Su sonda bajo 5.14.1 nunca corrio, asi que no hay evidencia de que falle ahi.
# Acotado a menciones cercanas a "qwen3.5-2b": una busqueda de substring sin
# acotar tambien matcheaba la evidencia legitima y ya verificada de que
# granite-4.0-350m "falla bajo 5.14.1" (linea 57 de 06_amenazas.tex), que no
# tiene nada que ver con Qwen3.5-2B y no es una sobreafirmacion.
for sobreafirmacion in ("falla bajo 5.14.1", "incompatible con 5.14.1",
                        "no corre bajo ninguna", "falla en las dos versiones"):
    patron = re.compile(
        rf"qwen3\.5-2b[^.]{{0,200}}{re.escape(sobreafirmacion)}"
        rf"|{re.escape(sobreafirmacion)}[^.]{{0,200}}qwen3\.5-2b"
    )
    assert not patron.search(texto), (
        f"sobreafirmacion sobre Qwen3.5-2B: {sobreafirmacion!r} cerca de su mencion. "
        f"Su compatibilidad bajo 5.14.1 es NO VERIFICADA, no demostradamente mala."
    )
print("las 3 amenazas de RF16 estan presentes (con las dos causas de exclusion), "
      "sin afirmacion de raw completion y sin sobreafirmar sobre Qwen3.5-2B")
PY
```
**Expected:** imprime `las 3 amenazas de RF16 estan presentes (con las dos causas de exclusion), sin afirmacion de raw completion y sin sobreafirmar sobre Qwen3.5-2B`, exit 0.
**On failure indicates:** el paper omite una limitación conocida del diseño, o resucitó la amenaza de incomparabilidad por raw completion que el Delta 02 eliminó porque los 12 del roster activo usan `chat_template`, o **sobreafirma** sobre `Qwen3.5-2B`. La tercera amenaza (exclusión por causas ajenas al método) es la más fácil de dejar incompleta: el Delta 02 la introdujo con **2** modelos por acceso no otorgado, y el Delta 2026-08-04 le sumó un **tercero** por compatibilidad **no verificada** — son **3 de 15**, con dos causas distintas que hay que atribuir bien. Y siguen siendo **tres** amenazas nuevas: si alguien agrega una cuarta, este check y RF16 dejan de coincidir.

### Check C12 — Determinismo del juez
**Covers AC:** AC5 (*reejecutar la etapa 2 sobre el mismo `detalle_2026.csv` reproduce `etiquetas_errores.csv` y `categorias_comandos.csv` byte a byte*).
**Cost:** `expensive` — reejecuta `src/judge_2026.py` dentro de la imagen del juez: carga un modelo de hasta 2B en CPU y hace inferencia sobre todas las respuestas incorrectas más los 32 comandos. Del orden de decenas de minutos. **No incluir en `--scope=cheap`.**
**Run:**
```bash
SLUG=$(python -c "import sys,json;sys.path.insert(0,'src');from models_2026 import slug;print(slug(json.load(open('data/2026/juez_seleccionado.json',encoding='utf-8'))['modelo']))")
md5sum data/2026/etiquetas_errores.csv data/2026/categorias_comandos.csv > /tmp/juez_antes.txt
docker run --rm --memory=8g --cpus=2 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  "slm-domotica-2026:${SLUG}" python src/judge_2026.py >/dev/null
md5sum -c /tmp/juez_antes.txt && echo "juez determinista OK"
```
**Expected:** `md5sum -c` imprime `OK` para los dos archivos y luego `juez determinista OK`.
**On failure indicates:** la etapa 2 no es reproducible — se coló `do_sample=True`, una `temperature` distinta de `None`, o una semilla/orden de iteración no determinista (RNF3). Un juez no determinista invalida las dos métricas titulares y el fallback de parseo congelado de F5.

### Check C13 — Correspondencia versión↔CSV (Delta 02, F5)
**Covers AC:** el invariante nuevo de F5 (*en cada CSV por modelo, `transformers_version` es constante y compatible con el `transformers_pin` de ese modelo: `4.57.*` para el grupo A, `5.*` para el grupo B*).
**Cost:** `cheap` — lecturas de CSV con pandas. Se salta si todavía no hay ningún CSV (antes del subtask 05).
**Run:**
```bash
python - <<'PY'
import glob
from pathlib import Path
import sys
sys.path.insert(0, "src")
import pandas as pd
from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo, slug

rutas = sorted(glob.glob("data/2026/detalle/*.csv"))
if not rutas:
    print("sin CSV todavia: C13 no aplica (requiere el subtask 05)")
    raise SystemExit(0)

fallas = []
for m in roster_activo():
    ruta = Path(f"data/2026/detalle/{slug(m.nombre)}.csv")
    if not ruta.exists():
        continue
    version = pd.read_csv(ruta)["transformers_version"].iloc[0]
    if m.transformers_pin == BASELINE_TRANSFORMERS and not str(version).startswith("4.57."):
        fallas.append(f"{m.nombre}: grupo A pero transformers_version={version}")
    if m.transformers_pin == TRANSFORMERS_5X and not str(version).startswith("5."):
        fallas.append(f"{m.nombre}: grupo B pero transformers_version={version}")
assert not fallas, fallas
print(f"correspondencia version-pin OK para {len(rutas)} CSV")
PY
```
**Expected:** `sin CSV todavia: C13 no aplica (requiere el subtask 05)` antes del barrido, o `correspondencia version-pin OK para 12 CSV` después: **9** con `4.57.*` (grupo A) y **3** con `5.*` (grupo B), tras el reparto del subtask 17. Exit 0 en ambos casos.
**On failure indicates:** un CSV se produjo bajo una versión de `transformers` distinta de la fijada para ese modelo — por ejemplo, `granite-4.0-350m` corrido bajo 5.x por error. Ese CSV es inválido y debe rehacerse (ver protocolo de fallo del subtask 05). Un caso concreto a vigilar tras el Delta 2026-08-04: `data/2026/detalle/qwen2-5-1-5b-instruct.csv` es el CSV del modelo que entró, y su `transformers_version` **tiene que** empezar con `4.57.` — su pin es del grupo A, sondeado (`PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template`), no inferido de la familia Qwen (que resuelve a grupo B en Qwen3.5).

### Check C14 — Exclusiones declaradas, con sus tres causas (Deltas 02, 2026-08-04 y 05)
**Covers AC:** AC13 (*los **cuatro** modelos excluidos no aparecen en ninguna tabla ni figura de resultados, y sí aparecen en la tabla de exclusiones de `docker/README.md` y en el texto del paper como limitación, con sus **tres causas distintas** correctamente atribuidas*), RF18.
**Nota sobre el conteo (Delta 05).** `Qwen2.5-0.5B-Instruct` se sumó al registro (15 → 16) para completar el brazo de generación anterior del baseline original (`baseline_original=True`, `activo=False`); es una tercera causa de exclusión (baseline-completion, de alcance de roster), no una regresión de las dos causas ya cubiertas (acceso no otorgado / compatibilidad no verificada). El roster activo de 12 no cambia.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
import re
from pathlib import Path
import sys
sys.path.insert(0, "src")
from models_2026 import MODELOS_2026

excluidos = [m for m in MODELOS_2026 if not m.activo]
assert len(excluidos) == 4, [m.nombre for m in excluidos]
assert {m.nombre for m in excluidos} == {
    "gemma-3-270m-it", "Llama-3.2-1B-Instruct", "Qwen3.5-2B", "Qwen2.5-0.5B-Instruct",
}, [m.nombre for m in excluidos]
# Tres causas distintas: 2 por acceso no otorgado (gated), 1 por compatibilidad no
# verificada (Qwen3.5-2B), 1 por baseline-completion (Qwen2.5-0.5B-Instruct, Delta 05).
assert len([m for m in excluidos if m.gated]) == 2
assert len([m for m in excluidos if not m.gated]) == 2
assert len([m for m in excluidos if m.baseline_original]) == 1

readme = Path("docker/README.md").read_text(encoding="utf-8")
fallas = [f"{m.nombre}: falta en la tabla de exclusiones de docker/README.md"
          for m in excluidos if m.nombre not in readme]
if "403" not in readme:
    fallas.append("docker/README.md: falta el 403 de los dos excluidos gated")
if "no verificada" not in readme.lower():
    fallas.append("docker/README.md: falta la causa 'compatibilidad no verificada' de Qwen3.5-2B")
if "baseline-completion" not in readme.lower():
    fallas.append("docker/README.md: falta la causa 'baseline-completion' de Qwen2.5-0.5B-Instruct")

for archivo in ("paper/02_reescrito/tablas/tabla1_modelos.tex",
                "paper/02_reescrito/tablas/tabla2_resultados_globales.tex",
                "paper/02_reescrito/tablas/tabla5_versiones.tex"):
    ruta = Path(archivo)
    if not ruta.exists():
        continue
    texto = ruta.read_text(encoding="utf-8")
    fallas += [f"{m.nombre}: aparece en {archivo} pese a estar excluido"
               for m in excluidos if m.nombre in texto]

for seccion in ("03_metodologia.tex", "06_amenazas.tex"):
    ruta = Path(f"paper/02_reescrito/secciones/{seccion}")
    if not ruta.exists():
        continue
    texto = ruta.read_text(encoding="utf-8").lower()
    if "acceso restringido" not in texto and "gated" not in texto:
        fallas.append(f"{seccion}: no declara la exclusion por acceso como limitacion")
    if "no verificada" not in texto:
        fallas.append(f"{seccion}: no declara la exclusion por compatibilidad no verificada")
    # Verificacion NEGATIVA: no se puede afirmar que Qwen3.5-2B falle bajo 5.14.1.
    # No hay evidencia de eso: la sonda nunca corrio. Solo ausencia de evidencia.
    # Acotado a menciones cercanas a "qwen3.5-2b" (ver misma nota en C11): un
    # substring sin acotar tambien matchea la evidencia legitima y ya verificada
    # de que granite-4.0-350m "falla bajo 5.14.1", que no es sobre Qwen3.5-2B.
    for sobreafirmacion in ("falla bajo 5.14.1", "incompatible con 5.14.1",
                            "no corre bajo ninguna", "falla en las dos versiones"):
        patron = re.compile(
            rf"qwen3\.5-2b[^.]{{0,200}}{re.escape(sobreafirmacion)}"
            rf"|{re.escape(sobreafirmacion)}[^.]{{0,200}}qwen3\.5-2b"
        )
        if patron.search(texto):
            fallas.append(f"{seccion}: sobreafirma sobre Qwen3.5-2B ({sobreafirmacion!r})")

assert not fallas, fallas
print("exclusiones declaradas OK: 4 excluidos (2 por acceso, 1 por compatibilidad no "
      "verificada, 1 por baseline-completion), fuera de tablas, dentro de "
      "exclusiones/limitacion, sin sobreafirmar")
PY
```
**Expected:** imprime `exclusiones declaradas OK: 4 excluidos (2 por acceso, 1 por compatibilidad no verificada, 1 por baseline-completion), fuera de tablas, dentro de exclusiones/limitacion, sin sobreafirmar`, exit 0.
**On failure indicates:** o bien un modelo excluido se coló en una tabla de resultados (contaminando una comparación que no corrió), o bien la exclusión no está documentada como limitación, o bien el paper **sobreafirma** sobre `Qwen3.5-2B`. Esto último es el error más fácil de cometer y el más caro: su sonda bajo 5.14.1 **nunca corrió** por un bloqueo de infraestructura de Docker, así que no hay evidencia de que falle bajo esa versión — solo ausencia de evidencia de que funcione. Escribir "incompatible con las dos versiones mayores" sería una afirmación empírica sin respaldo en un paper. La verificación negativa está acotada a menciones cercanas a `Qwen3.5-2B`: un substring sin acotar también dispara sobre la evidencia legítima de que `granite-4.0-350m` "falla bajo 5.14.1", que no tiene relación con `Qwen3.5-2B`.

### Check C15 — Continuidad con la Tabla 2 publicada (RF19, Delta 2026-08-04)
**Covers AC:** AC15 (*las dos columnas de continuidad de la Tabla 2 existen, sus valores coinciden dígito a dígito con `data/resultados_experimento_resumen.json` para los **3** modelos presentes en ambos estudios y son `--` para los otros **9**; §4 reporta la comparación y §5 la interpreta, incluida cualquier divergencia; y el texto declara que no es una réplica*), RF19.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "src")
from models_2026 import por_nombre, roster_activo

# 1. Los 3 anclas, derivados del cruce entre el registro y el resumen publicado (F0).
publicado = json.loads(
    Path("data/resultados_experimento_resumen.json").read_text(encoding="utf-8")
)
por_pub = {f["modelo"]: f for f in publicado}
anclas = sorted({m.nombre for m in roster_activo()} & set(por_pub))
assert anclas == ["Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct",
                  "SmolLM2-360M-Instruct"], anclas
for n in anclas:
    assert por_nombre(n).params_b == por_pub[n]["params_b"], n

fallas = []

# 2. La Tabla 2 trae las dos columnas y los valores publicados salen del JSON de F0.
tabla2 = Path("paper/02_reescrito/tablas/tabla2_resultados_globales.tex")
if tabla2.exists():
    tex = tabla2.read_text(encoding="utf-8")
    for etiqueta in ("Estricta 2025", "Latencia 2025"):
        if etiqueta not in tex:
            fallas.append(f"tabla2: falta la columna {etiqueta!r}")
    for n in anclas:
        estricta = f"{por_pub[n]['exact_match_pct']:.1f}"
        if estricta not in tex and estricta.replace(".", ",") not in tex:
            fallas.append(f"tabla2: falta la estricta publicada de {n} ({estricta})")
    # 9 de las 12 filas no tienen ancla publicada: sus dos celdas son '--'.
    sin_ancla = [m.nombre for m in roster_activo() if m.nombre not in por_pub]
    assert len(sin_ancla) == 9, sin_ancla
    if tex.count("--") < 2 * len(sin_ancla):
        fallas.append("tabla2: faltan celdas '--' para los 9 modelos sin ancla publicada")

# 3. §4 reporta la comparacion nombrando los 3 anclas; §5 la interpreta y aclara
#    que NO es una replica.
res = Path("paper/02_reescrito/secciones/04_resultados.tex")
if res.exists():
    t = res.read_text(encoding="utf-8")
    for n in anclas:
        if n not in t:
            fallas.append(f"04_resultados.tex: no nombra el ancla {n}")

dis = Path("paper/02_reescrito/secciones/05_discusion.tex")
if dis.exists():
    t = dis.read_text(encoding="utf-8").lower()
    if not any(k in t for k in ("divergenc", "diferencia con el estudio", "respecto de lo publicado")):
        fallas.append("05_discusion.tex: no discute la comparacion contra lo publicado")
    if not any(k in t for k in ("no es una replica", "no es una réplica",
                                "no es directamente comparable")):
        fallas.append("05_discusion.tex: no declara la salvedad de que no es una replica")

assert not fallas, fallas
print("continuidad OK: 3 anclas publicadas, 9 filas con '--', comparacion reportada y discutida")
PY
```
**Expected:** imprime `continuidad OK: 3 anclas publicadas, 9 filas con '--', comparacion reportada y discutida`, exit 0. Antes del subtask 11 los bloques 2 y 3 se saltean porque los archivos no existen todavía; el bloque 1 aplica desde el subtask 17.
**On failure indicates:** se perdió el único mecanismo de **validación cruzada del harness nuevo contra un resultado publicado**. Si faltan las columnas, la comparación quedó como prosa a mano (viola AC9: los números del paper vienen de fragmentos generados). Si falta la salvedad de §5, el paper presenta como réplica algo que no lo es: el prompt se endureció (RF1), así que la exactitud estricta no es directamente comparable, y la latencia arrastra además el confusor de los dos grupos de versión.

---

## 5. Non-functional checks

### NF1 — Envelope de recursos del barrido (comparabilidad de la latencia)
**Covers AC:** RNF1 (*misma máquina, CPU-only, `--memory=8g --cpus=2 --cpuset-cpus`, sin GPU, para preservar la comparabilidad de la tabla de latencia con el paper original*), RNF6 (*pinning de núcleos y ejecución secuencial*), AC17.
**Cost:** `cheap`
**Measurement / threshold:**
```bash
python docker/run_sweep.py --dry-run | grep -cE 'gpus|--device|nvidia' ; test $? -eq 1 && echo "sin GPU OK"
# build_all.py --dry-run imprime cada tag DOS veces (el header "=== build <tag> | <pin> ==="
# y la linea del comando "docker build ... -t <tag> ..."), asi que contar lineas que
# mencionan el tag da 24, no 12. Se cuentan tags DISTINTOS para que el numero refleje
# imagenes de verdad, no apariciones de texto.
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u | wc -l
# Reparto por grupo de version en el plan de build: 9 del grupo A, 3 del grupo B.
# Mismo problema de duplicacion que el tag (arriba): "transformers>=X" aparece
# en la linea de cabecera Y en la de "docker build --build-arg ...", asi que
# contar lineas sin mas da 18/6, el doble del real. Se dedupea igual que el
# tag: se extrae el par "<tag> | <pin>" (solo la linea de cabecera calza ese
# patron; la linea --build-arg no) y se cuenta sobre las lineas unicas.
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+ \| transformers[A-Za-z0-9.,<>=]+' | sort -u | grep -c 'transformers>=4.57.0,<5.0.0'
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+ \| transformers[A-Za-z0-9.,<>=]+' | sort -u | grep -c 'transformers>=5.0.0'
# Pinning de nucleos (Delta 2026-08-04): presente en las 12 y con UN SOLO valor.
python docker/run_sweep.py --dry-run | grep -c -- '--cpuset-cpus'
python docker/run_sweep.py --dry-run | grep -oE -- '--cpuset-cpus=[^ ]+' | sort -u | wc -l
python docker/run_sweep.py --dry-run | grep -c -- '--memory=8g'
python docker/run_sweep.py --dry-run | grep -c -- '--cpus=2'
# El valor efectivo queda registrado en la documentacion, no solo en el codigo.
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "docker"); sys.path.insert(0, "src")
from run_sweep import CPUSET_POR_DEFECTO
readme = Path("docker/README.md").read_text(encoding="utf-8")
assert f"--cpuset-cpus={CPUSET_POR_DEFECTO}" in readme, (
    f"docker/README.md no registra el --cpuset-cpus usado ({CPUSET_POR_DEFECTO})"
)
print(f"cpuset registrado en docker/README.md OK: {CPUSET_POR_DEFECTO}")
PY
```
**Expected:** `sin GPU OK` (cero menciones de GPU en las invocaciones), `12` imágenes **distintas** en el plan de build (roster activo), `9` y `3` para los dos pines, `12` invocaciones con `--cpuset-cpus`, **`1`** valor distinto de `--cpuset-cpus` (el mismo par de núcleos en las doce), `12` × `--memory=8g`, `12` × `--cpus=2`, y `cpuset registrado en docker/README.md OK: 0-1`. Esto fija el envelope completo de las 12 corridas.
**On failure indicates:** una corrida con recursos distintos hace que su latencia no sea comparable ni con las otras 11 ni con el paper original — rompe el eje derecho de fig1, la columna de latencia de la Tabla 2 y, con ella, la comparación de continuidad de C15. Si el conteo de valores distintos de `--cpuset-cpus` es mayor que 1, la comparabilidad se pierde de la peor forma posible: silenciosamente. Si el reparto no es 9/3, el intercambio de roster del subtask 17 no se aplicó. **Por qué el flag no es redundante con `--cpus=2`:** `--cpus` es una **cuota** del planificador CFS —permite migración entre núcleos y no reserva nada— mientras que `--cpuset-cpus` fija los núcleos y elimina la variabilidad por migración. Lo que ningún flag puede hacer es particionar la caché L3 ni el bus de memoria, y por eso la ejecución es **secuencial** (RNF6): correr 4 modelos en paralelo habría inflado los tiempos por comando de forma invisible en la tabla final.

### NF4 — Persistencia incremental y continuación ante fallo (RF20, F12)
**Covers AC:** AC16 (*un commit por modelo con el nombre y la versión efectiva de `transformers` en el asunto; `data/2026/fallos_barrido.json` existe, `[]` si no hubo fallos; ningún CSV parcial truncado*).
**Cost:** `cheap` — son lecturas de git y de CSV. **No** reejecuta el barrido.
**Measurement / threshold:**
```bash
python - <<'PY'
import glob
import json
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, "src")
import pandas as pd
from models_2026 import roster_activo, slug

csvs = sorted(glob.glob("data/2026/detalle/*.csv"))
if not csvs:
    print("sin CSV todavia: NF4 no aplica (requiere el subtask 05)")
    raise SystemExit(0)

fallas = []

# 1. Un commit por modelo, con la version efectiva en el asunto (F12.2).
asuntos = subprocess.run(
    ["git", "log", "--format=%s", "main..HEAD", "--grep=^data(2026): barrido de"],
    capture_output=True, text=True,
).stdout.splitlines()
if len(asuntos) != len(csvs):
    fallas.append(f"{len(asuntos)} commits por modelo para {len(csvs)} CSV")
for asunto in asuntos:
    if not re.match(r"^data\(2026\): barrido de .+ \(transformers \S+\)$", asunto):
        fallas.append(f"asunto sin la version efectiva: {asunto!r}")

# 2. El registro de fallos existe y tiene el esquema de F5.
ruta_fallos = Path("data/2026/fallos_barrido.json")
if not ruta_fallos.exists():
    fallas.append("falta data/2026/fallos_barrido.json (tiene que existir, [] si no hubo fallos)")
else:
    fallos = json.loads(ruta_fallos.read_text(encoding="utf-8"))
    assert isinstance(fallos, list)
    claves = {"modelo", "hf_repo_id", "transformers_pin",
              "codigo_salida", "error_textual", "momento_iso"}
    for f in fallos:
        if set(f) != claves:
            fallas.append(f"fallos_barrido.json: claves inesperadas {sorted(set(f))}")
    # F12.4: si hubo fallos, el conteo esperado ya no es 384.
    esperado = (len(roster_activo()) - len(fallos)) * 32
    print(f"fallos registrados: {len(fallos)} | filas esperadas del consolidado: {esperado}")

# 3. Ningun CSV parcial truncado: todos parsean y ninguno deja .tmp atras.
for ruta in csvs:
    try:
        pd.read_csv(ruta)
    except Exception as e:
        fallas.append(f"{ruta}: no parsea ({e})")
if glob.glob("data/2026/**/*.tmp", recursive=True):
    fallas.append("quedaron archivos .tmp: la escritura atomica no limpio")

assert not fallas, fallas
print(f"persistencia incremental OK: {len(csvs)} CSV, {len(asuntos)} commits por modelo")
PY
```
**Expected:** antes del subtask 05, `sin CSV todavia: NF4 no aplica (requiere el subtask 05)` y exit 0. Después: `fallos registrados: 0 | filas esperadas del consolidado: 384` y `persistencia incremental OK: 12 CSV, 12 commits por modelo`.
**On failure indicates:** si los commits por modelo son menos que los CSV, se acumularon resultados sin versionar y una caída del host habría costado horas de barrido — el modo de fallo concreto que RF20 vino a cerrar (esta sesión ya perdió trabajo tres veces: reinicio del backend de Docker, salida del proceso, error 529). Si falta `fallos_barrido.json`, no se puede distinguir "no hubo fallos" de "no se registró", y el paper no podría documentar una exclusión por fallo con su error textual. Si un CSV no parsea, la escritura no fue atómica y hay un parcial truncado haciéndose pasar por dato.

### NF2 — Reanudabilidad del barrido completo
**Covers AC:** RF3 / RNF2 (*barrido reanudable por modelo; reejecutar salta los modelos ya completos salvo `--force`; debe tolerar interrupciones*). Es cross-task: el subtask 03 verifica la reanudación de **un** modelo; acá se verifica que los **12** CSV del roster activo producidos por el subtask 05 sean reconocidos como completos.
**Cost:** `cheap` — corre el harness en el host, no en Docker. Como los 12 CSV ya existen, sale por la rama de *skip* sin cargar ni descargar ningún modelo. Segundos.
**Measurement / threshold:**
```bash
python - <<'PY'
import subprocess, sys
sys.path.insert(0, "src")
from models_2026 import roster_activo

fallas = []
for m in roster_activo():
    r = subprocess.run([sys.executable, "src/run_sweep_2026.py", "--modelo", m.nombre],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or "se saltea" not in r.stdout:
        fallas.append(m.nombre)
assert not fallas, f"no se reconocieron como completos: {fallas}"
print("los 12 modelos del roster activo se saltean: barrido reanudable OK")
PY
```
**Expected:** imprime `los 12 modelos del roster activo se saltean: barrido reanudable OK`. Cada invocación termina en menos de 120 s sin tocar la red.
**On failure indicates:** si un modelo **no** se saltea, una interrupción del barrido obligaría a reprocesarlo — o peor, el CSV de ese modelo está incompleto/corrupto y `run_sweep_2026.py` lo detecta. Si alguna invocación excede el timeout, está descargando el modelo: la lógica de reanudación no está mirando el CSV antes de cargar pesos, que es el orden que exige RF3.

### NF3 — Decodificación determinista en las dos etapas
**Covers AC:** RNF3 (*`do_sample=False`, `temperature=None`, `top_p=None` en las dos etapas*), RF7 (*greedy en etapa 1 y etapa 2*). Es el complemento estático y barato de C12.
**Cost:** `cheap`
**Measurement / threshold:**
```bash
git grep -nE 'do_sample\s*=\s*True|temperature\s*=\s*[0-9]' -- src/run_sweep_2026.py src/judge_2026.py \
  ; test $? -eq 1 && echo "sin muestreo estocastico OK"
git grep -cE 'do_sample\s*=\s*False' -- src/run_sweep_2026.py src/judge_2026.py
```
**Expected:** imprime `sin muestreo estocastico OK`, y el segundo comando lista **los dos** archivos, cada uno con al menos una ocurrencia (`src/judge_2026.py:1`, `src/run_sweep_2026.py:1` o más).
**On failure indicates:** una de las dos etapas muestrea. C12 lo detectaría también, pero después de decenas de minutos de inferencia; este check lo detecta en un segundo y por eso corre en cada iteración de fix.

---

## 6. Full regression sweep

### Tier `cheap` — se corre en **cada** iteración de fix (`/run-test-plan --scope=cheap`)

Ninguno de estos comandos arranca un contenedor de modelo, descarga pesos ni hace inferencia. Presupuesto total: **~3–5 min**, dominado por los dos builds de LaTeX.

| # | Run | Expected | Cost |
|---|-----|----------|------|
| 1 | `pytest -q` | exit 0, cero `failed` / `error` | `cheap` |
| 2 | C1 + C2 (inmutabilidad de F0, árbol e historial) | ambas salidas vacías | `cheap` |
| 3 | C3 + C4 + C5 (secretos, capas, untracked) | los cuatro mensajes `... OK` | `cheap` |
| 4 | C6 + NF1 (plomería del token y envelope de recursos, vía `--dry-run`) | `15`, `2`, `12`, `3`, `12`, `0`, `sin GPU OK`, `12`, `9`, `3`, `12`, `1`, `12`, `12`, `cpuset registrado ... 0-1` (registro de quince, dos gated, doce activos, tres excluidos, doce invocaciones, cero `--env-file`, doce imágenes **distintas** en el plan de build, reparto 9/3, pinning en las doce con **un solo** valor, y ese valor documentado) | `cheap` |
| 5 | NF3 (decodificación determinista) | `sin muestreo estocastico OK` | `cheap` |
| 6 | C7 (trazabilidad de versiones) | `... OK` | `cheap` |
| 7 | C8 (regeneración idempotente de tablas `.tex`) | `regeneracion idempotente OK` | `cheap` |
| 8 | C9 (compilan los dos papers) | dos líneas `compila OK` | `cheap` |
| 9 | C10 (envío ciego, después de C9) | `exit=0`, sin hallazgos | `cheap` |
| 10 | C11 (las 3 amenazas de RF16) | `las 3 amenazas de RF16 estan presentes, sin afirmacion de raw completion` | `cheap` |
| 11 | NF2 (reanudabilidad de los 12) | `barrido reanudable OK` | `cheap` |
| 12 | Manual Check 1 + Check 2 (figuras y PDF) | ver §3 | `cheap` |
| 13 | C13 (correspondencia versión↔CSV) | ver §4 | `cheap` |
| 14 | C14 (exclusiones declaradas, 3 con dos causas, sin sobreafirmar) | ver §4 | `cheap` |
| 15 | C15 (continuidad con la Tabla 2 publicada) | ver §4 | `cheap` |
| 16 | NF4 (persistencia incremental y registro de fallos) | ver §5 | `cheap` |

Los ítems 7–15 requieren que el DAG haya llegado al subtask 15 y que `data/2026/` exista; el ítem 16 requiere el subtask 05. Antes de eso, `/run-test-plan --scope=cheap` debe reportarlos como *no aplicables todavía*, no como fallos; los ítems 1–6 aplican desde el primer commit de la rama. **El ítem 4 aplica desde el primer commit pero sus números cambian con el subtask 17**: antes de que 17 esté cerrado, el registro es de 14, los excluidos son 2 y el reparto es 8/4, y **no hay** `--cpuset-cpus` — eso es esperado, no un fallo, hasta que 17 cierre. Después de 17, los valores de la tabla son los definitivos y cualquier desvío sí es un fallo.

### Tier `expensive` — solo en la **pasada confirmatoria**

Nunca en una iteración de fix. Requiere P4 (Docker up).

| # | Run | Expected | Cost |
|---|-----|----------|------|
| E1 | `python docker/build_all.py` | 12 imágenes `slm-domotica-2026:<slug>` del roster activo, **incluida** `slm-domotica-2026:qwen2-5-1-5b-instruct`; verificar con `comm -23` entre el plan (`build_all.py --dry-run`) y `docker images`, que debe dar vacío. **No** debe existir `slm-domotica-2026:qwen3-5-2b`: ese modelo está excluido | `expensive` (descargas de torch/transformers por imagen) |
| E2 | `python docker/run_sweep.py 2>&1 \| tee data/2026/log_barrido.txt` | 12 archivos en `data/2026/detalle/`, 384 filas en total, `fallos_barrido.json` = `[]`, y **un commit por modelo** (NF4). Si hubo fallos: `(12 − \|fallos\|) × 32` filas y los modelos fallidos documentados como limitación | `expensive` — **secuencial**, `--memory=8g --cpus=2 --cpuset-cpus=0-1`, CPU-only, ~7 h, host ocioso |
| E3 | C12 (determinismo del juez) | `juez determinista OK` | `expensive` (decenas de minutos de inferencia en CPU) |
| E4 | Rerun del tier `cheap` completo, sobre el estado post-E1/E2/E3 | todo verde, y en particular `pytest -q -rs tests/test_integracion_2026.py` con **cero** `skipped` | `cheap` |

Nota: como el roster activo tiene cero modelos *gated*, P4/E1/E2 ya **no** requieren `.env` ni `HF_TOKEN` para el roster de 12; esa plomería solo se ejercita si alguien reactiva explícitamente uno de los dos modelos excluidos **por acceso**. Reactivar `Qwen3.5-2B`, el tercer excluido, no requiere credenciales (no es *gated*) sino su sonda positiva bajo 5.14.1, que nunca se obtuvo.

**Nada en paralelo durante E2 (RNF6).** E2 corre **un modelo a la vez** y sobre **host ocioso**: no lanzar E1 de otras imágenes, ni E3, ni ningún check `cheap` que use Docker mientras E2 avanza. `--cpus=2` es una cuota de CFS y `--memory=8g` un techo, no reservas; `--cpuset-cpus` fija los núcleos pero ni la caché L3 ni el bus de memoria se pueden particionar por contenedor, y la inferencia de LLM en CPU está limitada por ancho de banda de memoria. Cualquier carga concurrente infla los tiempos por comando de forma **invisible** en la tabla final y contamina la columna de latencia, la comparación de continuidad de C15 y el eje derecho de fig1.

**E2 es reanudable y no se bloquea.** Si se interrumpe, `python docker/run_sweep.py --desde "<modelo>"` retoma y los modelos ya completos se saltean. Si un modelo falla, E2 **no corta**: registra el fallo en `data/2026/fallos_barrido.json` y sigue (F12.1). La única excepción es un 401/403, que corta inmediato (AUTH-STOP) y no se reintenta.

**Regla de oro del tier.** E2 **produce** los datos; AC2/AC3/AC4 los **verifican** y son `cheap`. Nunca reejecutar E1/E2 para revalidar un criterio de datos: si `data/2026/detalle_2026.csv` está en disco, el Test 3 lo verifica en segundos. La única razón legítima para volver a correr E2 es que el barrido en sí haya cambiado (roster, prompt, harness o imágenes) — y el intercambio de roster del subtask 17 **es** una de esas razones para cualquier CSV producido antes de él, salvo `data/2026/detalle/granite-4-0-350m.csv`, cuyo pin no cambió (F5).

**Precaución con `E2` + `tee`.** `data/2026/log_barrido.txt` es un archivo de log. Antes de commitearlo, C3 debe estar verde: un log de descarga puede contener el token en una URL firmada o en un traceback. Si C3 falla sobre el log, el remedio es no versionar el log **y rotar el token**.
