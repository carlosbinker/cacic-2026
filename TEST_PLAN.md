# Test plan — Reejecución del experimento con roster 2026, verificación automática en dos etapas y reescritura del paper en LaTeX

**Source TODO:** `TODO.md`
**Goal:** Probar que el DAG completo (subtasks 01–15 + G1/G2) produce un barrido de 14 modelos coherente de punta a punta, dos papers LaTeX que compilan y pasan el filtro de envío ciego, y un árbol versionado sin secretos ni regresiones sobre el material congelado de F0.

**Scope:** Solo checks **globales / cross-task**: los 12 criterios de §5 *Cross-task acceptance* del índice, más los invariantes del *Delta 01* (roster de 14 con exactamente 2 gated, plomería de `$HF_TOKEN` por `--env-file` acotada a esos 2, fallo temprano si falta el token, y cero secretos versionados). Fuera de alcance acá: todo lo que sea atribuible a un único subtask — vive en el bloque `Verify` de su `TODO_<NN>_<slug>.md` y **no** se reproduce en este archivo. También fuera de alcance lo que el índice declara *out of scope* en §2.4 (login interactivo, modelos `Qwen2.5-*`, abstract en inglés, chequeo del límite de 10 páginas, GPU/cuantización, juez externo por API).

Nota deliberada sobre solape: los criterios AC1–AC12 están declarados por el índice como *"se verifica al final del ciclo (`create-test-plan` / `run-test-plan`), no dentro de ningún subtask"*. Algunos subtasks hacen un *smoke check* del mismo hecho en el momento de su propio commit (p. ej. 15 compila los papers). Este plan los reejecuta como **compuerta de regresión sobre el estado final del árbol**, que es un hecho distinto del smoke check puntual: un fix posterior puede romperlos.

**Estructura de costo — leer antes de correr nada.** El tier caro de este proyecto **no es una API paga**: es el barrido de 14 modelos en Docker, CPU-only, `--memory=8g --cpus=2`, secuencial, estimado en **5–9 h** más **20–40 GB** de descargas, y la reejecución del juez. La distinción que gobierna todo el plan es:

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
Run: `docker info --format '{{.ServerVersion}}' && test -s .env && grep -q '^HF_TOKEN=.\+' .env && echo "docker + .env con HF_TOKEN: OK"`
Expected: imprime la versión del server de Docker y `docker + .env con HF_TOKEN: OK`. **Nunca imprimir el contenido de `.env`.** Si `.env` falta o `HF_TOKEN` está vacío, **parar**: los dos modelos gated no se pueden descargar y el barrido debe abortar temprano (eso es justamente lo que verifica T2).

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
**Expected:** exit 0, todos los tests `passed`. En particular: con una raíz temporal sin `.env`, `validar_credenciales(gated(), tmp)` levanta `ValueError`; con `HF_TOKEN` vacío también; con la lista de los 12 no-gated **no** levanta nada aunque falte `.env`; y `_hf_token_de_env` no escribe nada en stdout.
**On failure indicates:** el barrido de 5–9 h podría abortar a mitad de la descarga del modelo 7 o del 14 en vez de fallar en el segundo 0 — exactamente el modo de fallo caro que el Delta 01 vino a cerrar. O peor: el mensaje de error filtra el token.

### Test 3 — Integridad de los artefactos del barrido y de la etapa 2
**Covers AC:** AC2 (*448 filas, 14 modelos, sin `idx` faltantes ni duplicados*), AC3 (*cobertura de la etapa 2*), AC4 (*coherencia estricta/laxa*). Lo crea la tarea global **G2**.
**Cost:** `cheap` — son lecturas de `data/2026/*.csv` y `resumen_2026.json` con pandas. **No** reejecuta el barrido.
**Run:** `pytest -q -rs tests/test_integracion_2026.py`
**Expected:** exit 0. Antes de que el subtask 09 haya corrido, se aceptan `skipped` con motivo explícito (los tests están guardados por `skipif` sobre la existencia de los artefactos). En la **pasada confirmatoria** (post-subtask 09): `4 passed`, **cero** `skipped` — un `skipped` ahí significa que falta un artefacto que debería existir.
**On failure indicates:** el consolidado perdió o duplicó filas (bug en `stage1_2026.py`), el juez no cubrió todas las respuestas incorrectas o etiquetó fuera del vocabulario cerrado (bug en `judge_2026.py` o en el fallback de parseo de F5), o la métrica laxa quedó por debajo de la estricta (bug en `metrics_2026.py`, viola la definición congelada de F6).

---

## 3. Manual checks

### Check 1 — Legibilidad de las figuras con 14 modelos
**Covers AC:** RF12 (*barras horizontales agrupadas por tier, nombres largos legibles, sin apiñamiento a 14 modelos*). El subtask 10 verifica por script que los PNG existen, pesan >20 KB y contienen 14 modelos separados 7/7 por tier; **no** puede verificar que se lean bien. Eso es irreductiblemente humano.
**Cost:** `cheap`
**Steps:**
1. Abrir `figures/2026/fig1_exactitud_latencia_2026.png` con un visor de imágenes al 100 % de zoom.
2. Leer las 14 etiquetas del eje Y de arriba a abajo.
3. Abrir `figures/2026/fig2_exactitud_por_campo_2026.png` y repetir.
**Expected observation:** los 14 nombres del roster (`LFM2.5-230M` … `Llama-3.2-1B-Instruct`) se leen completos, sin truncado con `…` ni superposición entre etiquetas vecinas; el bloque `sub-1B` (7 barras) está visualmente separado del bloque `1-2B` (7 barras); en el panel izquierdo de fig1 se distinguen dos series superpuestas (estricta y laxa) con leyenda.
**On failure indicates:** el layout de `generate_figures_2026.py` no escaló de 8 a 14 modelos — la figura no es publicable aunque el script salga con exit 0.

### Check 2 — El PDF reescrito renderiza tablas y figuras
**Covers AC:** AC7 (*los dos papers compilan*) y AC9 (*números del paper == números de los datos*) en su dimensión visual: `latexmk` puede salir con 0 y aun así producir tablas desbordadas o `??` en las referencias cruzadas.
**Cost:** `cheap`
**Steps:**
1. Abrir `paper/02_reescrito/main.pdf`.
2. Localizar las cinco tablas (`tab:modelos`, `tab:globales`, `tab:categorias`, `tab:taxonomia`, `tab:versiones`) y las dos figuras.
3. Buscar en el texto la cadena `??`.
**Expected observation:** las cinco tablas y las dos figuras aparecen, ninguna se sale del margen de la caja de texto LNCS, la tabla 1 lista 14 filas de modelos, y no hay ninguna ocurrencia de `??` (referencia cruzada rota) en todo el PDF.
**On failure indicates:** un fragmento `.tex` generado por `generate_tex_tables.py` es demasiado ancho para el formato LNCS, o un `\label`/`\ref` quedó desparejado entre secciones.

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
**Expected:** el `git log` da salida **vacía**; el `git rev-list --count` da **≥ 17** (los 15 subtasks del DAG + G1 + G2 de §6 del índice, un commit cada uno).
**On failure indicates:** si el `git log` no está vacío, un subtask tocó F0 y otro lo revirtió — el árbol final miente sobre lo que pasó. Si el conteo es < 17, algún subtask no dejó su propio commit y se violó RNF5 (*un commit por subtask*), lo que rompe la trazabilidad que pide AC11.

### Check C3 — Ningún token de HuggingFace versionado (árbol completo)
**Covers AC:** AC12 (*Ningún secreto versionado*) y punto 4 del Delta 01.
**Cost:** `cheap`
**Run:** `git grep -iE 'hf_[A-Za-z0-9]{20,}' ; test $? -eq 1 && echo "sin token versionado OK"`
**Expected:** imprime `sin token versionado OK` y nada más. `git grep` sale con 1 (sin coincidencias).
**Nota sobre el patrón — no relajarlo.** El patrón exige `hf_` + **20 o más** alfanuméricos seguidos porque un token real es `hf_` + ~34 alfanuméricos. La forma laxa `hf_[A-Za-z]` es **incorrecta**: matchea identificadores legítimos del diseño (`hf_repo_id` de F3, `.hf_cache` de F11, `HF_TOKEN`, `HF_HOME`) que **deben** seguir existiendo en el árbol. Si este check empieza a dar falsos positivos, el bug está en el patrón, no en el repo.
**On failure indicates:** el valor del token se filtró a un archivo trackeado — probablemente un log commiteado del barrido (`data/2026/log_barrido.txt`), el `docker/README.md`, o un `TODO_<NN>_*.md`. Es un incidente de seguridad: hay que rotar el token, no solo borrar la línea.

### Check C4 — El token nunca entra en una capa de imagen
**Covers AC:** AC12 (segunda mitad) y F11 (*prohibido `ARG`/`ENV HF_TOKEN`, `COPY .env`, `--build-arg`, login interactivo*). El subtask 04 testea esto sobre `docker/`; acá se extiende a **todo el árbol trackeado**, porque un `Dockerfile` puede aparecer fuera de `docker/` (hay uno en la raíz, legacy).
**Cost:** `cheap`
**Run:**
```bash
git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env|--build-arg[= ]*HF_TOKEN' ; test $? -eq 1 && echo "sin credenciales en capas OK"
git grep -niE 'huggingface-cli login|huggingface_hub\.login\(' ; test $? -eq 1 && echo "sin login interactivo OK"
```
**Expected:** imprime `sin credenciales en capas OK` y `sin login interactivo OK`, sin ninguna línea de coincidencia.
**On failure indicates:** el token quedaría horneado en una capa de imagen (recuperable por cualquiera que tenga la imagen) o el pipeline dejó de ser no interactivo — ambas cosas están explícitamente prohibidas por F11 y por §2.4.

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

### Check C6 — `--env-file` exactamente para los dos gated, y para nadie más
**Covers AC:** punto 3 del Delta 01 / RF4 / RF17 / F11 (*solo lo reciben las dos imágenes que lo necesitan; los demás corren sin credenciales*), y RNF1 (*`--memory=8g --cpus=2` en las 14 invocaciones*).
**Cost:** `cheap` — `--dry-run` solo imprime los comandos; no arranca ningún contenedor ni descarga nada.
**Run:**
```bash
python docker/run_sweep.py --dry-run | grep -c -- '--env-file'
python docker/run_sweep.py --dry-run | grep -c -- '--memory=8g'
python docker/run_sweep.py --dry-run | grep -c -- '--cpus=2'
python docker/run_sweep.py --dry-run | grep -- '--env-file' | grep -cE 'slm-domotica-2026:(gemma-3-270m-it|llama-3-2-1b-instruct)'
```
**Expected:** `2`, `14`, `14`, `2` respectivamente. Es decir: 14 invocaciones, todas con el envelope de recursos congelado, y las **únicas** dos que reciben `--env-file` son las de `gemma-3-270m-it` y `Llama-3.2-1B-Instruct`.
**On failure indicates:** si el primer conteo es >2, el token se está exponiendo a contenedores que no lo necesitan (superficie de ataque innecesaria). Si es <2, un modelo gated va a fallar con 401 a mitad del barrido. Si los conteos de recursos no dan 14, la tabla de latencia deja de ser comparable con el paper original (RNF1).

### Check C7 — Trazabilidad de versiones divergentes
**Covers AC:** AC6 (*todo modelo con `transformers_pin != BASELINE_TRANSFORMERS` aparece con su motivo en `docker/README.md` **y** en `tabla5_versiones.tex` **y** se menciona en `03_metodologia.tex` y en `06_amenazas.tex`*), RF5.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from models_2026 import MODELOS_2026, BASELINE_TRANSFORMERS

divergentes = [m for m in MODELOS_2026 if m.transformers_pin != BASELINE_TRANSFORMERS]
if not divergentes:
    print("sin pines divergentes: AC6 se cumple de forma vacua OK")
    raise SystemExit(0)
destinos = {
    "docker/README.md": Path("docker/README.md"),
    "tabla5_versiones.tex": Path("paper/02_reescrito/tablas/tabla5_versiones.tex"),
    "03_metodologia.tex": Path("paper/02_reescrito/secciones/03_metodologia.tex"),
    "06_amenazas.tex": Path("paper/02_reescrito/secciones/06_amenazas.tex"),
}
textos = {k: v.read_text(encoding="utf-8") for k, v in destinos.items()}
fallas = [
    f"{m.nombre} no aparece en {k}"
    for m in divergentes
    for k, texto in textos.items()
    if m.nombre not in texto
]
fallas += [
    f"{m.nombre}: motivo_pin no aparece en docker/README.md"
    for m in divergentes
    if m.motivo_pin and m.motivo_pin[:30] not in textos["docker/README.md"]
]
assert not fallas, fallas
print(f"trazabilidad OK para {len(divergentes)} pin(es) divergente(s)")
PY
```
**Expected:** imprime `sin pines divergentes: AC6 se cumple de forma vacua OK` (si el subtask 04 no necesitó divergir de `transformers>=4.57.0`) o `trazabilidad OK para N pin(es) divergente(s)`. Exit 0 en ambos casos.
**On failure indicates:** una divergencia de librería quedó sin documentar. Es un confusor directo de la tabla de latencia y una de las cuatro amenazas de RF16 — si no está escrita, el paper afirma una comparación que no puede sostener.

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

### Check C11 — Las cuatro amenazas nuevas de RF16 están escritas
**Covers AC:** AC10 (*`06_amenazas.tex` cubre explícitamente las cuatro amenazas de RF16*), RF18.
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
from pathlib import Path
texto = Path("paper/02_reescrito/secciones/06_amenazas.tex").read_text(encoding="utf-8").lower()
amenazas = {
    "auto-favorecimiento del juez": (("juez",), ("favorec", "sesgo")),
    "modelos base por raw completion": (("raw completion", "completion cruda"), ("base", "comparab")),
    "versiones de libreria como confusor de latencia": (("versi",), ("latencia",)),
    "reproducibilidad limitada por los modelos gated": (("gated",), ("licencia", "token")),
}
faltan = [
    nombre
    for nombre, (anclas, extras) in amenazas.items()
    if not any(a in texto for a in anclas) or not any(e in texto for e in extras)
]
assert not faltan, f"amenazas ausentes o incompletas: {faltan}"
print("las 4 amenazas de RF16 estan presentes")
PY
```
**Expected:** imprime `las 4 amenazas de RF16 estan presentes`, exit 0.
**On failure indicates:** el paper omite una limitación conocida del diseño. La cuarta (reproducibilidad limitada por los dos modelos gated) es la que introdujo el Delta 01 y es la más fácil de olvidar, porque no existía en la versión previa del plan.

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

---

## 5. Non-functional checks

### NF1 — Envelope de recursos del barrido (comparabilidad de la latencia)
**Covers AC:** RNF1 (*misma máquina, CPU-only, `--memory=8g --cpus=2`, sin GPU, para preservar la comparabilidad de la tabla de latencia con el paper original*).
**Cost:** `cheap`
**Measurement / threshold:**
```bash
python docker/run_sweep.py --dry-run | grep -cE 'gpus|--device|nvidia' ; test $? -eq 1 && echo "sin GPU OK"
python docker/build_all.py --dry-run | grep -c 'slm-domotica-2026:'
```
**Expected:** `sin GPU OK` (cero menciones de GPU en las invocaciones) y `14` imágenes en el plan de build. Combinado con C6 (`14` × `--memory=8g`, `14` × `--cpus=2`), esto fija el envelope de las 14 corridas.
**On failure indicates:** una corrida con recursos distintos hace que su latencia no sea comparable ni con las otras 13 ni con el paper original — rompe el eje derecho de fig1 y la columna de latencia de la Tabla 2.

### NF2 — Reanudabilidad del barrido completo
**Covers AC:** RF3 / RNF2 (*barrido reanudable por modelo; reejecutar salta los modelos ya completos salvo `--force`; debe tolerar interrupciones*). Es cross-task: el subtask 03 verifica la reanudación de **un** modelo; acá se verifica que los **14** CSV producidos por el subtask 05 sean reconocidos como completos.
**Cost:** `cheap` — corre el harness en el host, no en Docker. Como los 14 CSV ya existen, sale por la rama de *skip* sin cargar ni descargar ningún modelo. Segundos.
**Measurement / threshold:**
```bash
python - <<'PY'
import subprocess, sys
sys.path.insert(0, "src")
from models_2026 import MODELOS_2026

fallas = []
for m in MODELOS_2026:
    r = subprocess.run([sys.executable, "src/run_sweep_2026.py", "--modelo", m.nombre],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or "se saltea" not in r.stdout:
        fallas.append(m.nombre)
assert not fallas, f"no se reconocieron como completos: {fallas}"
print("los 14 modelos se saltean: barrido reanudable OK")
PY
```
**Expected:** imprime `los 14 modelos se saltean: barrido reanudable OK`. Cada invocación termina en menos de 120 s sin tocar la red.
**On failure indicates:** si un modelo **no** se saltea, una interrupción del barrido de 5–9 h obligaría a reprocesarlo — o peor, el CSV de ese modelo está incompleto/corrupto y `run_sweep_2026.py` lo detecta. Si alguna invocación excede el timeout, está descargando el modelo: la lógica de reanudación no está mirando el CSV antes de cargar pesos, que es el orden que exige RF3.

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
| 4 | C6 + NF1 (plomería del token y envelope de recursos, vía `--dry-run`) | `2`, `14`, `14`, `2`, `sin GPU OK`, `14` | `cheap` |
| 5 | NF3 (decodificación determinista) | `sin muestreo estocastico OK` | `cheap` |
| 6 | C7 (trazabilidad de versiones) | `... OK` | `cheap` |
| 7 | C8 (regeneración idempotente de tablas `.tex`) | `regeneracion idempotente OK` | `cheap` |
| 8 | C9 (compilan los dos papers) | dos líneas `compila OK` | `cheap` |
| 9 | C10 (envío ciego, después de C9) | `exit=0`, sin hallazgos | `cheap` |
| 10 | C11 (las 4 amenazas de RF16) | `las 4 amenazas de RF16 estan presentes` | `cheap` |
| 11 | NF2 (reanudabilidad de los 14) | `barrido reanudable OK` | `cheap` |
| 12 | Manual Check 1 + Check 2 (figuras y PDF) | ver §3 | `cheap` |

Los ítems 7–12 requieren que el DAG haya llegado al subtask 15 y que `data/2026/` exista. Antes de eso, `/run-test-plan --scope=cheap` debe reportarlos como *no aplicables todavía*, no como fallos; los ítems 1–6 aplican desde el primer commit de la rama.

### Tier `expensive` — solo en la **pasada confirmatoria**

Nunca en una iteración de fix. Requiere P4 (Docker up + `.env` con `HF_TOKEN`).

| # | Run | Expected | Cost |
|---|-----|----------|------|
| E1 | `python docker/build_all.py` | 14 imágenes `slm-domotica-2026:<slug>`; verificar con `docker images --format '{{.Repository}}:{{.Tag}}' \| grep -c '^slm-domotica-2026:'` → `14` | `expensive` (descargas de torch/transformers por imagen) |
| E2 | `python docker/run_sweep.py 2>&1 \| tee data/2026/log_barrido.txt` | 14 archivos en `data/2026/detalle/`, 448 filas en total | `expensive` — **5–9 h, 20–40 GB**; secuencial, `--memory=8g --cpus=2`, CPU-only |
| E3 | C12 (determinismo del juez) | `juez determinista OK` | `expensive` (decenas de minutos de inferencia en CPU) |
| E4 | Rerun del tier `cheap` completo, sobre el estado post-E1/E2/E3 | todo verde, y en particular `pytest -q -rs tests/test_integracion_2026.py` con **cero** `skipped` | `cheap` |

**Regla de oro del tier.** E2 **produce** los datos; AC2/AC3/AC4 los **verifican** y son `cheap`. Nunca reejecutar E1/E2 para revalidar un criterio de datos: si `data/2026/detalle_2026.csv` está en disco, el Test 3 lo verifica en segundos. La única razón legítima para volver a correr E2 es que el barrido en sí haya cambiado (roster, prompt, harness o imágenes).

**Precaución con `E2` + `tee`.** `data/2026/log_barrido.txt` es un archivo de log. Antes de commitearlo, C3 debe estar verde: un log de descarga puede contener el token en una URL firmada o en un traceback. Si C3 falla sobre el log, el remedio es no versionar el log **y rotar el token**.
