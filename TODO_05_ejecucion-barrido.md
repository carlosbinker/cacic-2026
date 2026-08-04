---
id: 05
title: Ejecución del barrido completo (12 modelos del roster activo)
depends_on: [04, 16, 17]
files:
  - data/2026/detalle/
  - data/2026/fallos_barrido.json
  - data/2026/log_barrido.txt
  - docker/README.md
---

## Spec

Ejecutar el barrido real: los **12 modelos del roster activo** (`models_2026.roster_activo()`), de
a uno, cada uno en su imagen, con `--memory=8g --cpus=2 --cpuset-cpus=0-1`, produciendo los 12
`data/2026/detalle/<slug>.csv` de 32 filas cada uno (**384** filas en total). El roster activo tiene
**cero** modelos *gated*: el barrido corre **sin** `.env` ni `$HF_TOKEN` (ver la nota de invocación en
la Tarea 2). Los **tres** modelos excluidos (`gemma-3-270m-it` y `Llama-3.2-1B-Instruct` por acceso de
descarga no otorgado; `Qwen3.5-2B` por compatibilidad de versión **no verificada**, decisión del
usuario del 2026-08-04) tienen `activo=False` y **no se corren**. Es la tarea de **datos** que
desbloquea toda la mitad de análisis y de paper del DAG.

**Precondición dura:** este subtask depende del **16** (pines por grupo de versión, `roster_activo()`)
y del **17** (intercambio de roster, `--cpuset-cpus`, continuación ante fallo). No arrancar el barrido
si el 17 no está cerrado: correr contra el roster viejo —con `Qwen3.5-2B` dentro y sin
`Qwen2.5-1.5B-Instruct`— invalidaría los datos y rompería la línea de continuidad con la Tabla 2
publicada (RF19).

No se escribe código nuevo: se ejecuta el pipeline de los subtasks 03, 04, 16 y 17, y se commitean los
resultados **de a un modelo** (RF20a, F12.2). Duración esperada: ~7 h de reloj, secuencial (**RNF2**).
El harness es reanudable, así que una interrupción se retoma con `--desde`.

**Política de ejecución (RNF6, F11, F12) — no negociable:**

- **Un modelo a la vez.** Nunca dos contenedores de modelo en paralelo. El razonamiento completo está
  en RNF6 del índice y tiene que llegar a la metodología del paper (subtask 14): `--cpus` es una
  **cuota** de CFS y `--memory` un **techo**, ninguno es una reserva; `--cpuset-cpus` sí fija núcleos,
  pero **ni la caché L3 ni el bus de memoria se pueden particionar por contenedor**, y la inferencia de
  LLM en CPU está limitada por ancho de banda de memoria. Correr 4 modelos en paralelo habría inflado
  los tiempos por comando de forma **invisible** en la tabla final. Se eligió secuencial para que la
  latencia siga siendo comparable con el paper original, al costo de ~7 h en lugar de ~2 h.
- **Mismo `--cpuset-cpus` en las 12 corridas** (y en la del juez, subtask 08). Lo que hace comparable
  la latencia no es qué par de núcleos se use, sino que sea el mismo en todas. Valor por defecto:
  `0-1`. Si se cambia, se cambia para las 12 y queda registrado en `docker/README.md`.
- **Host ocioso.** Nada más corriendo en la máquina mientras el barrido avanza.
- **Persistir y commitear al obtener cada resultado** (RF20a): el CSV de cada modelo se commitea en
  cuanto ese modelo termina, no al final. Ver Tarea 2.
- **Continuar ante fallo** (RF20c, F12.1): si un modelo falla de forma irrecuperable, se registra en
  `data/2026/fallos_barrido.json` con su error textual y **el barrido sigue con el siguiente**. Ver el
  Protocolo de fallo. La **única** excepción es un 401/403: eso corta inmediato (AUTH-STOP).

**Regla de datos de sondeo.** Los tiempos de **todas** las sondas de compatibilidad
(`.claude-scratch/logs/probe.log`, `probe5x.log`, `sweep2.log` de Phase 3, y
`probe_qwen25_15b_groupA.log` del 2026-08-04) **no son datos de latencia** y no
pueden usarse, citarse ni aparecer en `data/2026/**`, en ninguna tabla o figura generada, ni en
ninguna afirmación de latencia del paper. La única fuente legítima de latencia es este barrido: las
32 corridas completas por modelo, en host ocioso, bajo `--cpus=2 --memory=8g`.

Si un modelo falla de forma irrecuperable, **no** se lo silencia ni se lo elimina del registro por
cuenta propia, pero **tampoco se detiene el barrido**: se registra el fallo con su error textual en
`data/2026/fallos_barrido.json` y se continúa con el siguiente modelo (RF20c). La exclusión resultante
se documenta en el paper como limitación (ver "Protocolo de fallo").

## Implementation plan

> Exención de TDD completa: esta tarea no agrega comportamiento, ejecuta el que ya está testeado en 03 y 04. Su corrección se verifica por invariantes sobre los datos producidos.

### Tarea 1 — Preparación

- [ ] **Gate bloqueante — Docker Desktop con almacenamiento en modo escritura.** El 2026-08-03 el
  daemon quedó con su almacenamiento en solo lectura a mitad de ronda de sondeo:
  ```
  mkdir /var/lib/docker/overlay2/...-init: read-only file system
  write /var/lib/docker/buildkit/snapshots.db: read-only file system
  ```
  En ese estado ninguna imagen se puede construir ni correr (un contenedor puede incluso figurar
  `Up` mientras el mismo daemon responde `container ... is not running`). El 2026-08-04 el daemon ya
  volvió a estado de escritura —la sonda de `Qwen2.5-1.5B-Instruct` corrió dentro de
  `slm-domotica-2026:granite-4-0-350m`— pero eso no lo garantiza para esta corrida. Confirmar antes de
  arrancar:
  ```bash
  docker info --format '{{.ServerVersion}}'
  docker run --rm slm-domotica-2026:granite-4-0-350m python -c "print('escritura y run OK')"
  ```
  Si el error de solo lectura reaparece: **parar**. Reparar Docker Desktop (reiniciarlo o
  recuperar/agrandar su disco virtual) es una decisión del usuario, que está usando la máquina; no se
  intenta desde acá.

> **Gate eliminado (Delta 2026-08-04).** Acá había un *gate bloqueante* que exigía la confirmación
> positiva de `Qwen3.5-2B` bajo `transformers` 5.14.1 antes de arrancar el barrido. **Queda
> eliminado**, y con él cualquier criterio de aceptación que dependiera de esa sonda. Razón: el
> usuario decidió sacar `Qwen3.5-2B` del roster activo **sin condición** (§2.1 del índice), así que la
> sonda ya no es una precondición de nada; y como el usuario **no está disponible**, un gate que
> requería su decisión habría detenido el barrido a mitad de camino sin nadie para desbloquearlo — que
> es exactamente el modo de fallo que este ciclo tiene que evitar. `Qwen3.5-2B` queda en el registro
> con `activo=False` y `motivo_exclusion` no vacío (subtask 17), y su exclusión se documenta en el
> paper como limitación, con la redacción prudente que corresponde: **compatibilidad no verificada**,
> nunca "incompatible con las dos versiones mayores". No queda ninguna referencia pendiente a esa sonda.

- [ ] **Confirmar que el subtask 17 está cerrado** (intercambio de roster, `--cpuset-cpus`,
  continuación ante fallo). Sin esto el barrido correría contra el roster viejo:
  ```bash
  pytest -q tests/test_models_2026.py tests/test_docker_matriz.py tests/test_credenciales_gated.py
  python -c "
  import sys; sys.path.insert(0,'src')
  from models_2026 import MODELOS_2026, por_nombre, roster_activo
  assert len(MODELOS_2026) == 15, len(MODELOS_2026)
  assert len(roster_activo()) == 12
  assert por_nombre('Qwen3.5-2B').activo is False, 'Qwen3.5-2B tiene que estar excluido'
  assert por_nombre('Qwen2.5-1.5B-Instruct').activo is True, 'falta Qwen2.5-1.5B-Instruct'
  print('roster del Delta 03 OK: registro 15, activo 12, intercambio aplicado')
  "
  ```
- [ ] **Confirmar que existe la imagen del modelo que entró** (la construyó el subtask 17, Tarea 7):
  `docker images --format '{{.Repository}}:{{.Tag}}' | grep '^slm-domotica-2026:qwen2-5-1-5b-instruct$'`
  Si no está, **parar** y cerrar la Tarea 7 del subtask 17 antes de seguir.
- [ ] **Confirmar que el roster activo no necesita credenciales.** El roster activo
  (`roster_activo()`) tiene cero modelos `gated`, así que este barrido **no** requiere `.env` ni
  `$HF_TOKEN`:
  ```bash
  python -c "
  import sys; sys.path.insert(0,'src')
  from models_2026 import roster_activo
  assert not any(m.gated for m in roster_activo()), 'el roster activo no deberia tener gated'
  print('roster activo sin gated: OK, no hace falta .env')
  "
  ```
  Si el barrido se corre alguna vez con un modelo reactivado (`activo=True` sobre `gemma-3-270m-it` o
  `Llama-3.2-1B-Instruct`), retoma la precondición de `.env`/`HF_TOKEN` del *Delta 01* (ver
  `docker/README.md`, sección de credenciales) antes de arrancar. Reactivar `Qwen3.5-2B`, en cambio,
  **no** requiere credenciales (no es gated) pero sí su sonda positiva bajo 5.14.1.
- [ ] Confirmar que las **3** imágenes del grupo B ya fueron reconstruidas con su pin y que la imagen
  nueva del grupo A existe (ver `TODO_17_intercambio-roster-y-ejecucion.md`, Tarea 7). La imagen de
  `Qwen3.5-2B` **no** debe existir ni construirse.
- [ ] Confirmar que el DAG previo está verde: `pytest -q`
- [ ] Confirmar que están las 12 imágenes del roster activo, sin faltar ninguna del plan:

  ```bash
  python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u > /tmp/plan.txt
  docker images --format '{{.Repository}}:{{.Tag}}' | grep '^slm-domotica-2026:' | sort -u > /tmp/hay.txt
  wc -l < /tmp/plan.txt          # -> 12
  comm -23 /tmp/plan.txt /tmp/hay.txt   # -> vacio: no falta ninguna imagen del plan
  ```
- [ ] Confirmar el envelope de recursos del plan, incluido el pinning (RNF1, RNF6):

  ```bash
  python docker/run_sweep.py --dry-run | grep -c -- '--cpuset-cpus'                  # -> 12
  python docker/run_sweep.py --dry-run | grep -oE -- '--cpuset-cpus=[^ ]+' | sort -u # -> una sola linea
  python docker/run_sweep.py --dry-run | grep -c -- '--memory=8g'                    # -> 12
  python docker/run_sweep.py --dry-run | grep -c -- '--cpus=2'                        # -> 12
  ```
- [ ] Confirmar que el host está ocioso: cerrar todo lo que consuma CPU antes de arrancar. La latencia
  es el resultado que esta condición protege.
- [ ] Confirmar espacio libre ≥ 50 GB en `C:` (pesos + capas de imagen, escalado a 12 modelos):
  `df -h /c | tail -1`
- [ ] Crear la caché compartida si no existe: `mkdir -p .hf_cache data/2026/detalle`
- [ ] Confirmar que la caché de pesos ya está ignorada (la entrada la escribió el subtask 01,
  único dueño de `.gitignore`; **no editarlo acá**):
  `git check-ignore -v .hf_cache/` → debe imprimir la regla que la ignora.
  Si no imprime nada, **parar** y completar el subtask 01 antes de seguir: un barrido con
  la caché sin ignorar mete decenas de GB en el índice.

### Tarea 2 — Barrido

> Nota (documentación, no código nuevo): `docker/run_sweep.py` corre, por defecto, las **12 imágenes
> del roster activo** (`roster_activo()`), de a una, en orden de registro:
> ```
> docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
>   -v <repo>/data:/app/data \
>   -v <repo>/.hf_cache:/app/.hf_cache \
>   slm-domotica-2026:<slug> \
>   python src/run_sweep_2026.py --modelo "<nombre>"
> ```
> Ninguna de las 12 lleva `--env-file`: el roster activo no tiene modelos *gated*. La invocación con
> `--env-file <repo>/.env` sigue **definida** en `docker/run_sweep.py` (F11) para los **dos** modelos
> *gated* del registro, pero este barrido no la usa; `Qwen3.5-2B` está excluido y **no** es gated, así
> que tampoco la usaría. Esta lógica ya está implementada en los subtasks 04, 16 y 17; acá solo se
> documenta para quien ejecute el barrido.

- [ ] Lanzar el barrido completo, con log persistente:

```bash
python docker/run_sweep.py 2>&1 | tee data/2026/log_barrido.txt
```

- [ ] **Commitear cada resultado en cuanto sale** (RF20a, F12.2). No esperar a los 12: lo que solo
  existe en el working tree es exactamente lo que cuesta una caída, y esta sesión ya perdió trabajo
  tres veces (reinicio del backend de Docker, salida del proceso, error 529). El mensaje lleva el
  nombre del modelo y su versión **efectiva** de `transformers`, la que realmente corrió dentro del
  contenedor (columna `transformers_version` de ese mismo CSV, no la pedida). Correr este bucle en otra
  terminal mientras el barrido avanza, o después de cada modelo:

```bash
python - <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, "src")
import pandas as pd
from models_2026 import roster_activo, slug

for m in roster_activo():
    ruta = Path(f"data/2026/detalle/{slug(m.nombre)}.csv")
    if not ruta.exists():
        continue
    # Ya commiteado? git no lo lista como modificado ni sin trackear.
    pendiente = subprocess.run(["git", "status", "--porcelain", "--", str(ruta)],
                               capture_output=True, text=True).stdout.strip()
    if not pendiente:
        continue
    version = pd.read_csv(ruta)["transformers_version"].iloc[0]
    subprocess.run(["git", "add", str(ruta)], check=True)          # F1: git add acotado
    subprocess.run(["git", "commit", "-m",
                    f"data(2026): barrido de {m.nombre} (transformers {version})"], check=True)
    print(f"commiteado {ruta.name} (transformers {version})")
PY
```

- [ ] Si se interrumpe, retomar desde el modelo que quedó a medias (los ya completos se saltean solos):

```bash
python docker/run_sweep.py --desde "<nombre del modelo>" 2>&1 | tee -a data/2026/log_barrido.txt
```

- [ ] Chequear el avance en cualquier momento (sobre el **roster activo**, no sobre el registro: los 3
  excluidos no producen CSV y contarlos como faltantes sería un falso positivo):

```bash
ls -1 data/2026/detalle/*.csv | wc -l          # cuántos modelos completos, esperado 12 al final
python -c "
import sys, glob; sys.path.insert(0,'src')
from models_2026 import roster_activo, slug
hechos = {p.replace('\\\\','/').split('/')[-1][:-4] for p in glob.glob('data/2026/detalle/*.csv')}
faltan = [m.nombre for m in roster_activo() if slug(m.nombre) not in hechos]
sobran = sorted(hechos - {slug(m.nombre) for m in roster_activo()})
print('faltan:', faltan or 'ninguno')
print('sobran (no son del roster activo):', sobran or 'ninguno')
"
```

- [ ] Al terminar, revisar el registro de fallos (**siempre existe**, `[]` si no hubo ninguno):

```bash
python -c "
import json
fallos = json.load(open('data/2026/fallos_barrido.json', encoding='utf-8'))
print(f'{len(fallos)} fallo(s)')
for f in fallos:
    print(f\"  {f['modelo']} | codigo {f['codigo_salida']} | {f['momento_iso']}\")
    print(f\"    {f['error_textual'][:300]}\")
"
```

  Si la lista **no** está vacía: el consolidado del subtask 06 tendrá `(12 − |fallos|) × 32` filas en
  vez de 384, y ese número tiene que quedar escrito en el commit del 06 y en el paper como limitación
  (F12.4). **No** se reintenta indefinidamente ni se falsean datos: se documenta.

### Tarea 3 — Validación de integridad del barrido

- [ ] Correr el chequeo de invariantes sobre los 12 CSV del roster activo:

```bash
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import roster_activo, slug
from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO
problemas = []
for m in roster_activo():
    ruta = f'data/2026/detalle/{slug(m.nombre)}.csv'
    try:
        df = pd.read_csv(ruta)
    except Exception as e:
        problemas.append(f'{m.nombre}: ilegible ({e})'); continue
    if list(df.columns) != COLUMNAS_DETALLE:
        problemas.append(f'{m.nombre}: esquema distinto')
    if len(df) != N_COMANDOS_ESPERADO:
        problemas.append(f'{m.nombre}: {len(df)} filas')
    if df['idx'].tolist() != list(range(N_COMANDOS_ESPERADO)):
        problemas.append(f'{m.nombre}: idx con huecos o desordenado')
    if df['modelo'].nunique() != 1 or df['modelo'].iloc[0] != m.nombre:
        problemas.append(f'{m.nombre}: columna modelo inconsistente')
    if df['latencia_s'].le(0).any():
        problemas.append(f'{m.nombre}: latencia no positiva')
print('PROBLEMAS:', problemas or 'ninguno')
assert not problemas
"
```

- [ ] Verificar que el modo de prompting detectado coincide con lo esperado en §2.1 del índice, y **si no coincide, corregir la tabla del índice** (gana la detección, no la tabla):

```bash
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import roster_activo, slug
for m in roster_activo():
    df = pd.read_csv(f'data/2026/detalle/{slug(m.nombre)}.csv')
    modos = df['modo_prompting'].unique().tolist()
    assert len(modos) == 1, (m.nombre, modos)
    print(f'{m.nombre:26s} {modos[0]:16s} transformers={df[\"transformers_version\"].iloc[0]}')
"
```

- [ ] Confirmar que los 12 modelos del roster activo salieron por `chat_template` (la corrección de
  Phase 3 estableció que los dos LFM2.5 base **sí** tienen `chat_template`; ninguno del roster activo
  ejercita `raw_completion`). Si alguno difiere, anotarlo: es un hallazgo a reportar en la
  metodología del paper, no un error, y **no** una razón para reintroducir una salvedad de
  comparabilidad que ya no aplica.
- [ ] Verificar la **correspondencia versión↔CSV** (invariante nuevo de F5): para cada modelo, la
  columna `transformers_version` del CSV es compatible con su `transformers_pin` (grupo A →
  `4.57.*`, grupo B → `5.*`):

```bash
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo, slug
problemas = []
for m in roster_activo():
    df = pd.read_csv(f'data/2026/detalle/{slug(m.nombre)}.csv')
    version = df['transformers_version'].iloc[0]
    if m.transformers_pin == BASELINE_TRANSFORMERS and not str(version).startswith('4.57.'):
        problemas.append(f'{m.nombre}: pin grupo A pero transformers_version={version}')
    if m.transformers_pin == TRANSFORMERS_5X and not str(version).startswith('5.'):
        problemas.append(f'{m.nombre}: pin grupo B pero transformers_version={version}')
print('PROBLEMAS de correspondencia version-pin:', problemas or 'ninguno')
assert not problemas
"
```

  Si `data/2026/detalle/granite-4-0-350m.csv` ya existe de una corrida previa bajo 4.57.6 (subtask
  04) y el operador no puede atestiguar que se produjo en host ocioso bajo `--cpus=2 --memory=8g`,
  rehacerlo con `--force`: `python docker/run_sweep.py --desde "granite-4.0-350m" --force`. Si sí
  puede atestiguarlo, **conservarlo tal cual** — no recomputar sin necesidad.

### Tarea 4 — Registrar el entorno efectivo y commitear

- [ ] Agregar al final de `docker/README.md` una sección con las versiones **realmente observadas** (no las pedidas), tomadas de la columna `transformers_version`, y el `--cpuset-cpus` usado:

```markdown
## Versiones efectivas del barrido

Registradas de la columna `transformers_version` de cada CSV de detalle,
es decir la versión que efectivamente corrió dentro del contenedor. Envelope
de la corrida: `--memory=8g --cpus=2 --cpuset-cpus=0-1`, secuencial, host ocioso.

| Modelo | `transformers` efectiva | Modo de prompting |
|---|---|---|
| ... una fila por cada uno de los 12 modelos del roster activo ... |
```

- [ ] `git add data/2026/log_barrido.txt data/2026/fallos_barrido.json docker/README.md`
- [ ] Commit de cierre del subtask (los 12 CSV ya fueron commiteados de a uno en la Tarea 2, RF20a):
  ```
  git commit -m "data(2026): cierre del barrido de los 12 modelos (384 corridas, CPU 2 nucleos pinneados)"
  ```
  Si hubo fallos, el asunto dice el conteo real en vez de 384, y el cuerpo lista los modelos fallidos
  con su código de salida.
- [ ] Confirmar que quedó **un commit por modelo**, además del de cierre:
  ```bash
  git log --oneline main..HEAD --grep='^data(2026): barrido de' | wc -l   # -> igual a la cantidad de CSV
  ls -1 data/2026/detalle/*.csv | wc -l
  ```

### Protocolo de fallo

Si un modelo falla y no se recupera tras **un** reintento:

- [ ] El error textual completo queda en `data/2026/log_barrido.txt` **y** en
  `data/2026/fallos_barrido.json` (lo escribe `docker/run_sweep.py`, F12.1). Verificar que esté; si no
  está, es un bug del subtask 17.
- [ ] Si es un problema de versión de librería → **no** se ajusta el pin sobre la marcha. Se registra,
  el barrido continúa con el siguiente modelo, y el ajuste de `transformers_pin`/`trust_remote_code`
  con su `motivo_pin` es una tarea correctiva posterior sobre `src/models_2026.py` (el patrón del
  subtask 17), con la imagen rehecha y el modelo retomado con `--desde`. Cambiar un pin a mitad del
  barrido rompería la correspondencia versión↔CSV de F5 para ese modelo.
- [ ] Si es un fallo del modelo en sí (arquitectura no soportada, pesos rotos) → **NO se detiene el
  barrido** (RF20c). El usuario no está disponible para desbloquearlo y los 11 restantes son datos que
  se perderían. Se registra el fallo con su error textual, se sigue, y **la exclusión se documenta en
  el paper como limitación** (subtasks 14/15), con la misma redacción prudente que `Qwen3.5-2B`: se
  dice qué falló y con qué error, no se especula sobre por qué. Consecuencias que hay que arrastrar
  explícitamente (F12.4): el consolidado del subtask 06 tiene `(12 − |fallos|) × 32` filas en vez de
  384, la Tabla 1 y la Tabla 2 tienen menos filas, y ese número aparece en el commit del 06 y en el
  paper. Lo que **no** se hace: silenciar el fallo, inventar datos, ni borrar el modelo del registro.
- [ ] Si es un error de autenticación / 401 / 403 → **PARAR inmediatamente**, no reintentar, no
  ejecutar ningún comando de login/auth/configure, y reportar `AUTH-BLOCKER` con el error textual
  completo. **Esta es la única excepción a "continuar ante fallo".** El roster activo no tiene modelos
  *gated*, así que **ningún** modelo de este barrido debería requerir credenciales: un 401/403 acá
  indica que algo se desvió gravemente del plan (por ejemplo, que se coló un modelo excluido). No se
  toca `.env`.

## Verify

```bash
# 1. Hay 12 CSV, uno por modelo del roster activo, y ninguno de más
ls -1 data/2026/detalle/*.csv | wc -l    # -> 12

# 2. 384 filas en total, 32 por modelo, sin duplicados (modelo, idx)
python -c "
import sys, glob; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import roster_activo, slug
dfs = [pd.read_csv(p) for p in sorted(glob.glob('data/2026/detalle/*.csv'))]
todo = pd.concat(dfs, ignore_index=True)
assert len(todo) == 384, len(todo)
assert todo.groupby('modelo').size().eq(32).all()
assert not todo.duplicated(['modelo','idx']).any()
assert set(todo['modelo']) == {m.nombre for m in roster_activo()}
print('384 filas OK, 12 modelos, sin duplicados')
"

# 3. Los porcentajes de exactitud estricta son plausibles y no todos iguales
python -c "
import sys, glob; sys.path.insert(0,'src')
import pandas as pd
todo = pd.concat([pd.read_csv(p) for p in glob.glob('data/2026/detalle/*.csv')])
r = (100*todo.groupby('modelo')['match_exact'].mean()).round(1).sort_values()
print(r.to_string())
assert r.between(0,100).all()
assert r.nunique() > 1, 'todos los modelos idénticos: sospechoso'
"

# 4. Los 12 del roster activo salieron por chat_template (ninguno ejercita raw_completion)
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import roster_activo, slug
for m in roster_activo():
    df = pd.read_csv(f'data/2026/detalle/{slug(m.nombre)}.csv')
    print(m.nombre, df['modo_prompting'].unique())
"

# 5. Correspondencia version-pin: transformers_version compatible con transformers_pin (F5)
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo, slug
for m in roster_activo():
    df = pd.read_csv(f'data/2026/detalle/{slug(m.nombre)}.csv')
    v = df['transformers_version'].iloc[0]
    if m.transformers_pin == BASELINE_TRANSFORMERS:
        assert str(v).startswith('4.57.'), (m.nombre, v)
    else:
        assert str(v).startswith('5.'), (m.nombre, v)
print('correspondencia version-pin OK')
"

# 6. Ningun modelo excluido produjo CSV (los 3 de activo=False no se corrieron)
python -c "
import sys, glob; sys.path.insert(0,'src')
from models_2026 import MODELOS_2026, slug
hechos = {p.replace('\\\\','/').split('/')[-1][:-4] for p in glob.glob('data/2026/detalle/*.csv')}
excluidos = [m.nombre for m in MODELOS_2026 if not m.activo]
assert len(excluidos) == 3, excluidos
colados = [n for n in excluidos if slug(n) in hechos]
assert not colados, f'se colaron modelos excluidos: {colados}'
print('ningun excluido produjo CSV OK (3 excluidos:', ', '.join(excluidos) + ')')
"

# 7. Registro de fallos presente (siempre existe; [] si no hubo)
python -c "
import json
fallos = json.load(open('data/2026/fallos_barrido.json', encoding='utf-8'))
assert isinstance(fallos, list)
claves = {'modelo','hf_repo_id','transformers_pin','codigo_salida','error_textual','momento_iso'}
for f in fallos:
    assert set(f) == claves, set(f)
print(f'fallos_barrido.json OK: {len(fallos)} fallo(s)')
"

# 8. Un commit por modelo, con la version efectiva en el asunto (RF20a, F12.2)
test "$(git log --oneline main..HEAD --grep='^data(2026): barrido de' | wc -l)" \
     -eq "$(ls -1 data/2026/detalle/*.csv | wc -l)" \
  && echo "un commit por modelo OK"
git log --format='%s' main..HEAD --grep='^data(2026): barrido de' | grep -cv 'transformers ' \
  ; test $? -eq 1 && echo "todos los asuntos llevan la version efectiva OK"

# 9. Los archivos congelados siguen intactos y los tests verdes
git diff --exit-code main -- data/resultados_experimento_detalle.csv \
  data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv \
  tests/test_metricas.py && echo "F0 intacto"
pytest -q
```

## Acceptance criteria

- **Dado** el inicio de la ejecución, **entonces** ningún modelo del roster activo requiere `.env` ni `HF_TOKEN` (cero `gated` en `roster_activo()`); el barrido arranca sin credenciales.
- **Dado** `data/2026/detalle/`, **entonces** contiene exactamente 12 archivos `.csv`, cuyos nombres son los `slug` de los 12 modelos del roster activo, sin sobrantes ni faltantes.
- **Dado** cada CSV, **entonces** tiene exactamente 32 filas, columnas idénticas y en el orden de `COLUMNAS_DETALLE` (17), `idx` de 0 a 31 sin huecos ni repeticiones, un único valor en la columna `modelo` igual al nombre del roster, y todas las latencias estrictamente positivas.
- **Dado** el concatenado de los 12, **entonces** tiene 384 filas y ningún par `(modelo, idx)` duplicado.
- **Dado** cada CSV, **entonces** su columna `modo_prompting` tiene un único valor, y ese valor es `chat_template` para los 12 (ninguno del roster activo ejercita `raw_completion`); cualquier discrepancia con la tabla §2.1 del índice quedó **corregida en el índice**, no ocultada.
- **Dado** cada CSV, **entonces** su columna `transformers_version` tiene un único valor, **compatible con el grupo del `transformers_pin`** de ese modelo (grupo A → `4.57.*`, grupo B → `5.*`), y esas versiones efectivas están tabuladas en `docker/README.md`.
- **Dado** `data/2026/detalle/granite-4-0-350m.csv`, **entonces** o bien se conservó tal cual (si el operador puede atestiguar host ocioso bajo `--cpus=2 --memory=8g`) o bien se rehizo con `--force` bajo esas condiciones; en ambos casos su `transformers_version` empieza con `4.57.`.
- **Dado** el conjunto de exactitudes estrictas por modelo, **entonces** están en `[0, 100]` y no son todas idénticas (todas iguales indicaría que el barrido no varió realmente de modelo).
- **Dado** el barrido, **cuando** se interrumpió y se retomó, **entonces** los modelos ya completos no se recalcularon y el resultado final es indistinguible de una corrida sin interrupciones.
- **Dado** los **tres** modelos con `activo=False` (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`, `Qwen3.5-2B`), **entonces** ninguno produjo un CSV en `data/2026/detalle/` y ninguno se intentó correr.
- **Dado** las 12 invocaciones ejecutadas, **entonces** todas llevaron `--memory=8g`, `--cpus=2` y `--cpuset-cpus` con **el mismo** valor, corrieron **de a una** (nunca dos contenedores de modelo en simultáneo) sobre host ocioso, y ese valor de `--cpuset-cpus` quedó registrado en `docker/README.md` junto a las versiones efectivas.
- **Dado** cada modelo terminado, **entonces** su CSV se commiteó **antes** de arrancar el siguiente, con un asunto de la forma `data(2026): barrido de <nombre> (transformers <version efectiva>)` donde la versión es la de la columna `transformers_version` de ese CSV; y la cantidad de esos commits es igual a la cantidad de CSV en `data/2026/detalle/`.
- **Dado** el fin del barrido, **entonces** `data/2026/fallos_barrido.json` **existe** —con `[]` si no hubo fallos— y cada entrada tiene exactamente las claves `modelo`, `hf_repo_id`, `transformers_pin`, `codigo_salida`, `error_textual`, `momento_iso`.
- **Dado** un modelo que falló de forma irrecuperable (no autenticación), **entonces** el barrido **continuó** con los siguientes, el fallo quedó registrado con su error textual, el conteo esperado del consolidado pasó a `(12 − |fallos|) × 32` y quedó escrito en el commit de cierre, y la exclusión quedó anotada para documentarse como limitación en el paper — **no** se silenció, ni se inventaron datos, ni se borró el modelo del registro.
- **Dado** el repositorio tras el commit de cierre, **entonces** los archivos de F0 siguen byte-idénticos a `main`, `pytest -q` pasa, y `data/2026/log_barrido.txt` contiene el registro de la corrida.
- **Dado** cualquier fallo de autenticación durante el barrido, **entonces** el proceso se detuvo **inmediatamente** —única excepción a la continuación ante fallo— sin reintentar y sin ejecutar ningún comando de login/auth/configure, hecho que, dado que el roster activo no tiene modelos *gated*, indicaría por sí mismo una desviación grave del plan, y quedó reportado como `AUTH-BLOCKER`.
- **Dado** el texto de este subtask, **entonces** no queda **ninguna** referencia al gate bloqueante de la sonda de `Qwen3.5-2B` bajo 5.14.1 como precondición del barrido: fue eliminado por el Delta 2026-08-04 y ningún criterio de aceptación depende de él.
- **Dado** cualquier tabla, figura o afirmación de latencia producida a partir de este subtask, **entonces** proviene exclusivamente de las 384 corridas completas de este barrido, nunca de los tiempos de las sondas de compatibilidad de Phase 3.
