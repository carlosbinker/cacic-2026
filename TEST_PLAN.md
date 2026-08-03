# Test plan — Reejecución del experimento con roster 2026, verificación automática en dos etapas y reescritura del paper en LaTeX

**Source TODO:** `TODO.md`
**Goal:** Probar que el DAG completo (subtasks 01–16 + G1/G2, tras el Delta 2026-08-03) produce un barrido de 12 modelos del roster activo coherente de punta a punta, dos papers LaTeX que compilan y pasan el filtro de envío ciego, y un árbol versionado sin secretos ni regresiones sobre el material congelado de F0.

**Scope:** Solo checks **globales / cross-task**: los criterios de §5 *Cross-task acceptance* del índice (ya extendido a 14 puntos por el Delta 02), más los invariantes del *Delta 01* y del *Delta 02* (roster activo de 12 con cero gated, dos de 14 excluidos por acceso no otorgado, dos grupos de versión de `transformers` mutuamente excluyentes, plomería de `$HF_TOKEN` conservada pero no exigida por el roster activo, y cero secretos versionados). Fuera de alcance acá: todo lo que sea atribuible a un único subtask — vive en el bloque `Verify` de su `TODO_<NN>_<slug>.md` y **no** se reproduce en este archivo. También fuera de alcance lo que el índice declara *out of scope* en §2.4 (login interactivo, modelos `Qwen2.5-*`, abstract en inglés, chequeo del límite de 10 páginas, GPU/cuantización, juez externo por API).

Nota deliberada sobre solape: los criterios AC1–AC14 están declarados por el índice como *"se verifica al final del ciclo (`create-test-plan` / `run-test-plan`), no dentro de ningún subtask"*. Algunos subtasks hacen un *smoke check* del mismo hecho en el momento de su propio commit (p. ej. 15 compila los papers). Este plan los reejecuta como **compuerta de regresión sobre el estado final del árbol**, que es un hecho distinto del smoke check puntual: un fix posterior puede romperlos.

**Estructura de costo — leer antes de correr nada.** El tier caro de este proyecto **no es una API paga**: es el barrido de los 12 modelos del roster activo en Docker, CPU-only, `--memory=8g --cpus=2`, secuencial, estimado y escalado de las 5–9 h originales, más las descargas, y la reejecución del juez. La distinción que gobierna todo el plan es:

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
Expected: imprime la versión del server de Docker y `docker: OK`. Tras el Delta 02, el roster activo (12) tiene **cero** modelos *gated*, así que E1/E2 **no** requieren `.env` ni `HF_TOKEN`. Esa plomería sigue existiendo (T2 la sigue verificando como mecanismo) pero solo se ejerce si el barrido incluye explícitamente uno de los dos modelos excluidos — en ese caso, sí hace falta `.env` con `HF_TOKEN` no vacío antes de arrancar.

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
**Expected:** exit 0, todos los tests `passed`. En particular: con una raíz temporal sin `.env`, `validar_credenciales(gated(), tmp)` levanta `ValueError`; con `HF_TOKEN` vacío también; con la lista de los 12 no-gated (o, tras el subtask 16, con `roster_activo()`) **no** levanta nada aunque falte `.env`; y `_hf_token_de_env` no escribe nada en stdout.
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
**Expected observation:** los 12 nombres del roster activo (`LFM2.5-230M` … `SmolLM2-1.7B-Instruct`) se leen completos, sin truncado con `…` ni superposición entre etiquetas vecinas; el bloque `sub-1B` (6 barras) está visualmente separado del bloque `1-2B` (6 barras); en el panel izquierdo de fig1 se distinguen dos series superpuestas (estricta y laxa) con leyenda; ninguno de los dos modelos excluidos (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) aparece.
**On failure indicates:** el layout de `generate_figures_2026.py` no escaló de 8 a 12 modelos, o un modelo excluido se coló en el gráfico — la figura no es publicable aunque el script salga con exit 0.

### Check 2 — El PDF reescrito renderiza tablas y figuras
**Covers AC:** AC7 (*los dos papers compilan*) y AC9 (*números del paper == números de los datos*) en su dimensión visual: `latexmk` puede salir con 0 y aun así producir tablas desbordadas o `??` en las referencias cruzadas.
**Cost:** `cheap`
**Steps:**
1. Abrir `paper/02_reescrito/main.pdf`.
2. Localizar las cinco tablas (`tab:modelos`, `tab:globales`, `tab:categorias`, `tab:taxonomia`, `tab:versiones`) y las dos figuras.
3. Buscar en el texto la cadena `??`.
**Expected observation:** las cinco tablas y las dos figuras aparecen, ninguna se sale del margen de la caja de texto LNCS, la tabla 1 lista 12 filas de modelos (el roster activo, sin los 2 excluidos), y no hay ninguna ocurrencia de `??` (referencia cruzada rota) en todo el PDF.
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

### Check C6 — el barrido corre sin `--env-file`; pedir un excluido falla con el motivo
**Covers AC:** RF4 / RF17 / F11 tras el Delta 02 (*el plan por defecto son 12 invocaciones sobre el roster activo, ninguna con `--env-file`; pedir explícitamente un modelo excluido falla con su `motivo_exclusion`; la receta con `--env-file` sigue existiendo y se aplica solo a `gated()` cuando se la fuerza*), y RNF1 (*`--memory=8g --cpus=2` en las 12 invocaciones*).
**Cost:** `cheap` — `--dry-run` solo imprime los comandos; no arranca ningún contenedor ni descarga nada.
**Run:**
```bash
# Secuencia base (alimenta la fila 4 de la tabla de regresion junto con NF1):
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import gated; print(len(gated()))"
python -c "import sys; sys.path.insert(0,'src'); from models_2026 import roster_activo; print(len(roster_activo()))"
python docker/run_sweep.py --dry-run | grep -c '^==='
python docker/run_sweep.py --dry-run | grep -c -- '--env-file'
```
**Expected:** `2` (gated en el registro), `12` (roster activo), `12` (invocaciones planeadas), `0` (ninguna con `--env-file`).
**On failure indicates:** si el conteo de `--env-file` es mayor que 0, algo reactivó un modelo gated sin que el índice lo sepa (el token se expondría a un contenedor sin necesitarlo). Si los conteos de gated/activo no son 2/12, el registro o el roster activo se desalinearon del contrato de F3.

**Verificaciones adicionales (mismo check, no entran en la fila 4 de la tabla de regresión):**
```bash
# Pedir explicitamente un modelo excluido falla con el motivo
python docker/run_sweep.py --desde "gemma-3-270m-it" --dry-run ; echo "exit=$?"

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
**Expected (adicional):** exit distinto de 0 al pedir un excluido, con un mensaje que nombra el `motivo_exclusion` (403/acceso no otorgado); `receta --env-file conservada para los 2 gated OK`.
**On failure indicates (adicional):** si pedir un excluido no falla, el barrido podría intentar descargar un modelo sin acceso y romper a mitad de camino. Si la receta con `--env-file` desapareció, se perdió la plomería de credenciales que el Delta 01 introdujo — necesaria si algún modelo se reactiva en el futuro.

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
amenazas = {
    "auto-favorecimiento del juez": (("juez",), ("favorec", "sesgo")),
    "versiones de libreria como confusor de latencia": (("versi",), ("latencia",)),
    "exclusion por acceso restringido no otorgado": (("acceso restringido", "gated"), ("metodologica",)),
}
faltan = [
    nombre
    for nombre, (anclas, extras) in amenazas.items()
    if not any(a in texto for a in anclas) or not any(e in texto for e in extras)
]
assert not faltan, f"amenazas ausentes o incompletas: {faltan}"
# Verificacion NEGATIVA (Delta 02): no puede quedar ninguna afirmacion de que un
# modelo del roster se prompteo por raw completion; esa amenaza fue ELIMINADA.
prohibido = re.search(r"(lfm2\.5-230m|lfm2\.5-350m|modelos base)[^.]{0,80}raw completion", texto)
assert not prohibido, "sobrevive la amenaza eliminada de incomparabilidad por raw completion"
print("las 3 amenazas de RF16 estan presentes, sin afirmacion de raw completion")
PY
```
**Expected:** imprime `las 3 amenazas de RF16 estan presentes, sin afirmacion de raw completion`, exit 0.
**On failure indicates:** el paper omite una limitación conocida del diseño, o resucitó la amenaza de incomparabilidad por raw completion que el Delta 02 eliminó porque los 12 del roster activo usan `chat_template`. La tercera (exclusión por acceso restringido) es la que introdujo el Delta 02 y es la más fácil de olvidar.

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
**Expected:** `sin CSV todavia: C13 no aplica (requiere el subtask 05)` antes del barrido, o `correspondencia version-pin OK para N CSV` después. Exit 0 en ambos casos.
**On failure indicates:** un CSV se produjo bajo una versión de `transformers` distinta de la fijada para ese modelo — por ejemplo, `granite-4.0-350m` corrido bajo 5.x por error. Ese CSV es inválido y debe rehacerse (ver protocolo de fallo del subtask 05).

### Check C14 — Exclusiones declaradas (Delta 02)
**Covers AC:** el punto nuevo de §5 (*los dos modelos excluidos no aparecen en ninguna tabla ni figura de resultados, y sí aparecen en la tabla de exclusiones de `docker/README.md` y en el texto del paper como limitación*).
**Cost:** `cheap`
**Run:**
```bash
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "src")
from models_2026 import MODELOS_2026

excluidos = [m for m in MODELOS_2026 if not m.activo]
assert len(excluidos) == 2

readme = Path("docker/README.md").read_text(encoding="utf-8")
fallas = [f"{m.nombre}: falta en la tabla de exclusiones de docker/README.md"
          for m in excluidos if m.nombre not in readme]

for archivo in ("paper/02_reescrito/tablas/tabla1_modelos.tex",
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
        fallas.append(f"{seccion}: no declara la exclusion como limitacion")

assert not fallas, fallas
print("exclusiones declaradas OK: 2 excluidos, fuera de tablas, dentro de exclusiones/limitacion")
PY
```
**Expected:** imprime `exclusiones declaradas OK: 2 excluidos, fuera de tablas, dentro de exclusiones/limitacion`, exit 0.
**On failure indicates:** o bien un modelo excluido se coló en una tabla de resultados (contaminando una comparación que no corrió), o bien la exclusión no está documentada como limitación — ambos son errores de integridad del reporte.

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
**Expected:** `sin GPU OK` (cero menciones de GPU en las invocaciones) y `12` imágenes en el plan de build (roster activo). Combinado con C6 (`12` × `--memory=8g`, `12` × `--cpus=2`), esto fija el envelope de las 12 corridas.
**On failure indicates:** una corrida con recursos distintos hace que su latencia no sea comparable ni con las otras 11 ni con el paper original — rompe el eje derecho de fig1 y la columna de latencia de la Tabla 2.

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
| 4 | C6 + NF1 (plomería del token y envelope de recursos, vía `--dry-run`) | `2`, `12`, `12`, `0`, `sin GPU OK`, `12` (dos gated en el registro, doce activos, doce invocaciones, cero `--env-file`) | `cheap` |
| 5 | NF3 (decodificación determinista) | `sin muestreo estocastico OK` | `cheap` |
| 6 | C7 (trazabilidad de versiones) | `... OK` | `cheap` |
| 7 | C8 (regeneración idempotente de tablas `.tex`) | `regeneracion idempotente OK` | `cheap` |
| 8 | C9 (compilan los dos papers) | dos líneas `compila OK` | `cheap` |
| 9 | C10 (envío ciego, después de C9) | `exit=0`, sin hallazgos | `cheap` |
| 10 | C11 (las 3 amenazas de RF16) | `las 3 amenazas de RF16 estan presentes, sin afirmacion de raw completion` | `cheap` |
| 11 | NF2 (reanudabilidad de los 12) | `barrido reanudable OK` | `cheap` |
| 12 | Manual Check 1 + Check 2 (figuras y PDF) | ver §3 | `cheap` |
| 13 | C13 (correspondencia versión↔CSV) | ver §4 | `cheap` |
| 14 | C14 (exclusiones declaradas) | ver §4 | `cheap` |

Los ítems 7–14 requieren que el DAG haya llegado al subtask 15 y que `data/2026/` exista. Antes de eso, `/run-test-plan --scope=cheap` debe reportarlos como *no aplicables todavía*, no como fallos; los ítems 1–6 aplican desde el primer commit de la rama.

### Tier `expensive` — solo en la **pasada confirmatoria**

Nunca en una iteración de fix. Requiere P4 (Docker up).

| # | Run | Expected | Cost |
|---|-----|----------|------|
| E1 | `python docker/build_all.py` | 12 imágenes `slm-domotica-2026:<slug>` del roster activo; verificar con `docker images --format '{{.Repository}}:{{.Tag}}' \| grep -c '^slm-domotica-2026:'` → `>= 12` | `expensive` (descargas de torch/transformers por imagen) |
| E2 | `python docker/run_sweep.py 2>&1 \| tee data/2026/log_barrido.txt` | 12 archivos en `data/2026/detalle/`, 384 filas en total | `expensive` — secuencial, `--memory=8g --cpus=2`, CPU-only |
| E3 | C12 (determinismo del juez) | `juez determinista OK` | `expensive` (decenas de minutos de inferencia en CPU) |
| E4 | Rerun del tier `cheap` completo, sobre el estado post-E1/E2/E3 | todo verde, y en particular `pytest -q -rs tests/test_integracion_2026.py` con **cero** `skipped` | `cheap` |

Nota: como el roster activo tiene cero modelos *gated*, P4/E1/E2 ya **no** requieren `.env` ni `HF_TOKEN` para el roster de 12; esa plomería solo se ejercita si alguien reactiva explícitamente uno de los dos modelos excluidos.

**Regla de oro del tier.** E2 **produce** los datos; AC2/AC3/AC4 los **verifican** y son `cheap`. Nunca reejecutar E1/E2 para revalidar un criterio de datos: si `data/2026/detalle_2026.csv` está en disco, el Test 3 lo verifica en segundos. La única razón legítima para volver a correr E2 es que el barrido en sí haya cambiado (roster, prompt, harness o imágenes).

**Precaución con `E2` + `tee`.** `data/2026/log_barrido.txt` es un archivo de log. Antes de commitearlo, C3 debe estar verde: un log de descarga puede contener el token en una URL firmada o en un traceback. Si C3 falla sobre el log, el remedio es no versionar el log **y rotar el token**.
