---
id: 05
title: Ejecución del barrido completo (12 modelos del roster activo)
depends_on: [04, 16]
files:
  - data/2026/detalle/
  - docker/README.md
---

## Spec

Ejecutar el barrido real: los **12 modelos del roster activo** (`models_2026.roster_activo()`), de
a uno, cada uno en su imagen, con `--memory=8g --cpus=2`, produciendo los 12
`data/2026/detalle/<slug>.csv` de 32 filas cada uno (**384** filas en total). El roster activo tiene
**cero** modelos *gated*: el barrido corre **sin** `.env` ni `$HF_TOKEN` (ver la nota de invocación en
la Tarea 2). `gemma-3-270m-it` y `Llama-3.2-1B-Instruct` quedan fuera del roster activo (acceso de
descarga no otorgado, `activo=False`) y no se corren. Es la tarea de **datos** que desbloquea toda la
mitad de análisis y de paper del DAG.

**Precondición dura:** este subtask depende también del **16** (roster activo de 12, pines por grupo
de versión, imágenes del grupo B reconstruidas). No arrancar el barrido si el subtask 16 no está
cerrado: correr contra pines viejos o contra el roster de 14 invalidaría los datos.

No se escribe código nuevo: se ejecuta el pipeline de los subtasks 03, 04 y 16, y se commitean los
resultados. Duración esperada, escalada a 12 modelos (**RNF2**). El harness es reanudable, así que una
interrupción se retoma con `--desde`.

**Regla de datos de sondeo.** Los tiempos de las sondas de compatibilidad de Phase 3
(`.claude-scratch/logs/probe.log`, `probe5x.log`, `sweep2.log`) **no son datos de latencia** y no
pueden usarse, citarse ni aparecer en `data/2026/**`, en ninguna tabla o figura generada, ni en
ninguna afirmación de latencia del paper. La única fuente legítima de latencia es este barrido: las
32 corridas completas por modelo, en host ocioso, bajo `--cpus=2 --memory=8g`.

Si un modelo falla de forma irrecuperable, **no** se lo silencia ni se lo elimina del roster por cuenta propia: se detiene, se documenta el fallo y se escala (ver "Protocolo de fallo").

## Implementation plan

> Exención de TDD completa: esta tarea no agrega comportamiento, ejecuta el que ya está testeado en 03 y 04. Su corrección se verifica por invariantes sobre los datos producidos.

### Tarea 1 — Preparación

- [ ] **Gate bloqueante — Docker Desktop reparado y con almacenamiento en modo escritura.** El
  daemon quedó con su almacenamiento en solo lectura a mitad de ronda de sondeo:
  ```
  mkdir /var/lib/docker/overlay2/...-init: read-only file system
  write /var/lib/docker/buildkit/snapshots.db: read-only file system
  ```
  En ese estado ninguna imagen se puede construir ni correr (un contenedor puede incluso figurar
  `Up` mientras el mismo daemon responde `container ... is not running`). No arrancar el barrido
  hasta reiniciar Docker Desktop o recuperar/agrandar su disco virtual, y confirmar que un `docker
  run` trivial funciona.
- [ ] **Gate bloqueante — confirmación positiva de `Qwen3.5-2B` bajo 5.14.1.** Su sonda nunca llegó
  a correr (bloqueada por el gate anterior); todavía no hay evidencia positiva propia. Antes de
  arrancar el barrido:
  ```bash
  docker run --rm -v <repo>/.hf_cache:/app/.hf_cache \
    -v <repo>/.claude-scratch/probe.py:/app/probe.py \
    slm-domotica-2026:qwen3-5-2b python /app/probe.py "Qwen3.5-2B"
  ```
  Se espera `PROBE_OK|Qwen3.5-2B|5.*|chat_template`. Si esa sonda falla también bajo 5.14.1,
  `Qwen3.5-2B` no correría bajo ninguna versión mayor y **eso es una decisión del usuario** —
  sacarlo del roster activo (queda en 11) o abrir un tercer grupo de versión con su propio confusor
  declarado — nunca algo que se resuelva inventando un tercer grupo por cuenta propia.
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
  `docker/README.md`, sección de credenciales) antes de arrancar.
- [ ] Confirmar que el subtask 16 está cerrado: `pytest -q tests/test_models_2026.py
  tests/test_docker_matriz.py` verde, y que las 4 imágenes del grupo B ya fueron reconstruidas con
  su pin nuevo (ver `TODO_16_roster-12-y-pins.md`, Tarea 6).
- [ ] Confirmar que el DAG previo está verde: `pytest -q`
- [ ] Confirmar las 12 imágenes del roster activo: `docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^slm-domotica-2026:'` → **≥ 12** (pueden existir también las 2 imágenes definidas-pero-excluidas si alguien las construyó antes; lo que importa es que las 12 del roster activo estén presentes: `python docker/build_all.py --dry-run | grep -c 'build slm-domotica-2026:'` → `12`)
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
> docker run --rm --memory=8g --cpus=2 \
>   -v <repo>/data:/app/data \
>   -v <repo>/.hf_cache:/app/.hf_cache \
>   slm-domotica-2026:<slug> \
>   python src/run_sweep_2026.py --modelo "<nombre>"
> ```
> Ninguna de las 12 lleva `--env-file`: el roster activo no tiene modelos *gated*. La invocación con
> `--env-file <repo>/.env` sigue **definida** en `docker/run_sweep.py` (F11) para los dos modelos
> excluidos, pero este barrido no la usa. Esta lógica ya está implementada en los subtasks 04 y 16;
> acá solo se documenta para quien ejecute el barrido.

- [ ] Lanzar el barrido completo, con log persistente:

```bash
python docker/run_sweep.py 2>&1 | tee data/2026/log_barrido.txt
```

- [ ] Si se interrumpe, retomar desde el modelo que quedó a medias:

```bash
python docker/run_sweep.py --desde "<nombre del modelo>" 2>&1 | tee -a data/2026/log_barrido.txt
```

- [ ] Chequear el avance en cualquier momento:

```bash
ls -1 data/2026/detalle/*.csv | wc -l          # cuántos modelos completos
python -c "
import sys, glob; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import MODELOS_2026, slug
hechos = {p.split('/')[-1][:-4] for p in glob.glob('data/2026/detalle/*.csv')}
faltan = [m.nombre for m in MODELOS_2026 if slug(m.nombre) not in hechos]
print('faltan:', faltan or 'ninguno')
"
```

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

- [ ] Agregar al final de `docker/README.md` una sección con las versiones **realmente observadas** (no las pedidas), tomadas de la columna `transformers_version`:

```markdown
## Versiones efectivas del barrido

Registradas de la columna `transformers_version` de cada CSV de detalle,
es decir la versión que efectivamente corrió dentro del contenedor:

| Modelo | `transformers` efectiva | Modo de prompting |
|---|---|---|
| ... una fila por modelo ... |
```

- [ ] `git add data/2026/detalle/ data/2026/log_barrido.txt docker/README.md`
- [ ] `git commit -m "data(2026): barrido completo de los 12 modelos del roster activo (384 corridas, CPU 2 nucleos)"`

### Protocolo de fallo

Si un modelo falla y no se recupera tras un reintento:

- [ ] Guardar el error textual completo en `data/2026/log_barrido.txt`.
- [ ] Si es un problema de versión de librería → volver al subtask 16, ajustar `transformers_pin`/`trust_remote_code` con su `motivo_pin`, rehacer la imagen, retomar con `--desde`.
- [ ] Si es un fallo del modelo en sí (arquitectura no soportada en ninguna versión, pesos rotos) → **PARAR y escalar**. Sacar un modelo del roster activo cambia §2.1 del índice, la Tabla 1 del paper y el conteo de 384 filas: es una re-congelación de contrato, no una decisión de implementación.
- [ ] Si es un error de autenticación / 401 / 403 → **PARAR inmediatamente**, no reintentar, no ejecutar ningún comando de login/auth/configure, y reportar `AUTH-BLOCKER` con el error textual completo. El roster activo no tiene modelos *gated*, así que **ningún** modelo de este barrido debería requerir credenciales: un 401/403 acá indica que algo se desvió gravemente del plan (por ejemplo, que se coló un modelo excluido). No se reintenta ni se corre ningún comando de login; no se toca `.env`.

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

# 6. Los archivos congelados siguen intactos y los tests verdes
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
- **Dado** el repositorio tras el commit, **entonces** los archivos de F0 siguen byte-idénticos a `main`, `pytest -q` pasa, y `data/2026/log_barrido.txt` contiene el registro de la corrida.
- **Dado** cualquier fallo de autenticación durante el barrido, **entonces** el proceso se detuvo sin reintentar y sin ejecutar ningún comando de login/auth/configure — hecho que, dado que el roster activo no tiene modelos *gated*, indicaría por sí mismo una desviación grave del plan — y quedó reportado como `AUTH-BLOCKER`.
- **Dado** cualquier tabla, figura o afirmación de latencia producida a partir de este subtask, **entonces** proviene exclusivamente de las 384 corridas completas de este barrido, nunca de los tiempos de las sondas de compatibilidad de Phase 3.
