---
id: 05
title: Ejecución del barrido completo (14 modelos)
depends_on: [04]
files:
  - data/2026/detalle/
  - docker/README.md
---

## Spec

Ejecutar el barrido real: los 14 modelos, de a uno, cada uno en su imagen, con `--memory=8g --cpus=2`, produciendo los 14 `data/2026/detalle/<slug>.csv` de 32 filas cada uno (448 filas en total). Los dos modelos *gated* (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) corren con `--env-file .env` para inyectar `$HF_TOKEN`; los otros 12 corren sin credenciales (ver la nota de invocación en la Tarea 2). Es la tarea de **datos** que desbloquea toda la mitad de análisis y de paper del DAG.

No se escribe código nuevo: se ejecuta el pipeline de los subtasks 03 y 04 y se commitean los resultados. Duración esperada ~5–9 h (**RNF2**; rescalado de ~4–8 h para 12 modelos a 14), el harness es reanudable, así que una interrupción se retoma con `--desde`.

Si un modelo falla de forma irrecuperable, **no** se lo silencia ni se lo elimina del roster por cuenta propia: se detiene, se documenta el fallo y se escala (ver "Protocolo de fallo").

## Implementation plan

> Exención de TDD completa: esta tarea no agrega comportamiento, ejecuta el que ya está testeado en 03 y 04. Su corrección se verifica por invariantes sobre los datos producidos.

### Tarea 1 — Preparación

- [ ] **Precondición de credenciales (verificar primero, antes de cualquier otra cosa).** `.env`
  existe en la raíz del repo, define un `HF_TOKEN` no vacío, y sigue sin trackear. **No abrir ni
  imprimir su contenido**; usar solo chequeos que no revelen el valor:
  ```bash
  test -s .env && grep -q '^HF_TOKEN=.\+' .env && echo ".env con HF_TOKEN: OK"
  git ls-files .env   # debe salir vacío (untracked)
  ```
  Si falta `.env` o `HF_TOKEN` está ausente/vacío, **parar**: los dos modelos *gated*
  (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) no van a poder descargarse y el barrido fallaría a
  mitad de camino, no al arrancar.
- [ ] Verificar (**solo verificar; no editar** — el subtask 01 es el único dueño de `.gitignore`)
  que ya cubre `.env` y `*.env`:
  `git check-ignore -v .env` → debe imprimir la regla que lo ignora. Si no imprime nada, **parar**
  y resolverlo en el subtask 01 antes de seguir.
- [ ] Confirmar que el DAG previo está verde: `pytest -q`
- [ ] Confirmar las 14 imágenes: `docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^slm-domotica-2026:'` → `14`
- [ ] Confirmar espacio libre ≥ 60 GB en `C:` (pesos + capas de imagen):
  `df -h /c | tail -1`
- [ ] Crear la caché compartida si no existe: `mkdir -p .hf_cache data/2026/detalle`
- [ ] Confirmar que la caché de pesos ya está ignorada (la entrada la escribió el subtask 01,
  único dueño de `.gitignore`; **no editarlo acá**):
  `git check-ignore -v .hf_cache/` → debe imprimir la regla que la ignora.
  Si no imprime nada, **parar** y completar el subtask 01 antes de seguir: un barrido con
  la caché sin ignorar mete decenas de GB en el índice.

### Tarea 2 — Barrido

> Nota (documentación, no código nuevo): `docker/run_sweep.py` corre las 14 imágenes, de a una, en
> orden de roster. Para los 12 modelos no *gated* invoca:
> ```
> docker run --rm --memory=8g --cpus=2 \
>   -v <repo>/data:/app/data \
>   -v <repo>/.hf_cache:/app/.hf_cache \
>   slm-domotica-2026:<slug> \
>   python src/run_sweep_2026.py --modelo "<nombre>"
> ```
> Para los 2 modelos *gated* (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) agrega **exclusivamente**
> `--env-file <repo>/.env`:
> ```
> docker run --rm --memory=8g --cpus=2 \
>   --env-file <repo>/.env \
>   -v <repo>/data:/app/data \
>   -v <repo>/.hf_cache:/app/.hf_cache \
>   slm-domotica-2026:<slug> \
>   python src/run_sweep_2026.py --modelo "<nombre>"
> ```
> Esta lógica ya está implementada en el subtask 04; acá solo se documenta para quien ejecute el
> barrido. El valor de `$HF_TOKEN` no aparece en ningún log ni en este archivo.

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

- [ ] Correr el chequeo de invariantes sobre los 14 CSV:

```bash
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import MODELOS_2026, slug
from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO
problemas = []
for m in MODELOS_2026:
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
from models_2026 import MODELOS_2026, slug
for m in MODELOS_2026:
    df = pd.read_csv(f'data/2026/detalle/{slug(m.nombre)}.csv')
    modos = df['modo_prompting'].unique().tolist()
    assert len(modos) == 1, (m.nombre, modos)
    print(f'{m.nombre:26s} {modos[0]:16s} transformers={df[\"transformers_version\"].iloc[0]}')
"
```

- [ ] Confirmar que los dos LFM2.5 base salieron por `raw_completion` y los otros diez por `chat_template`. Si alguno difiere, anotarlo: es un hallazgo a reportar en la metodología del paper, no un error.

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
- [ ] `git commit -m "data(2026): barrido completo de los 14 modelos (448 corridas, CPU 2 nucleos)"`

### Protocolo de fallo

Si un modelo falla y no se recupera tras un reintento:

- [ ] Guardar el error textual completo en `data/2026/log_barrido.txt`.
- [ ] Si es un problema de versión de librería → volver al subtask 04, ajustar `transformers_pin`/`trust_remote_code` con su `motivo_pin`, rehacer la imagen, retomar con `--desde`.
- [ ] Si es un fallo del modelo en sí (arquitectura no soportada en ninguna versión, pesos rotos) → **PARAR y escalar**. Sacar un modelo del roster cambia §2.1 del índice, la Tabla 1 del paper y el conteo de 448 filas: es una re-congelación de contrato, no una decisión de implementación.
- [ ] Si es un error de autenticación / 401 / 403 → **PARAR inmediatamente**, no reintentar, no ejecutar ningún comando de login/auth/configure, y reportar `AUTH-BLOCKER` con el error textual completo. Si ocurre en uno de los 12 modelos **no** *gated*, algo se desvió gravemente del plan (ninguno de esos debería requerir credenciales). Si ocurre en uno de los dos modelos *gated* (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) pese a haber pasado la precondición de la Tarea 1, no se reintenta ni se corre ningún comando de login: se escala igual, sin volver a tocar `.env`.

## Verify

```bash
# 1. Hay 14 CSV, uno por modelo del roster, y ninguno de más
ls -1 data/2026/detalle/*.csv | wc -l    # -> 14

# 2. 448 filas en total, 32 por modelo, sin duplicados (modelo, idx)
python -c "
import sys, glob; sys.path.insert(0,'src')
import pandas as pd
from models_2026 import MODELOS_2026, slug
dfs = [pd.read_csv(p) for p in sorted(glob.glob('data/2026/detalle/*.csv'))]
todo = pd.concat(dfs, ignore_index=True)
assert len(todo) == 448, len(todo)
assert todo.groupby('modelo').size().eq(32).all()
assert not todo.duplicated(['modelo','idx']).any()
assert set(todo['modelo']) == {m.nombre for m in MODELOS_2026}
print('448 filas OK, 14 modelos, sin duplicados')
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

# 4. Los dos modelos base salieron por raw_completion
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
for s in ('lfm2-5-230m','lfm2-5-350m'):
    df = pd.read_csv(f'data/2026/detalle/{s}.csv')
    print(s, df['modo_prompting'].unique())
"

# 5. Los archivos congelados siguen intactos y los tests verdes
git diff --exit-code main -- data/resultados_experimento_detalle.csv \
  data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv \
  tests/test_metricas.py && echo "F0 intacto"
pytest -q
```

## Acceptance criteria

- **Dado** el inicio de la ejecución, **entonces** `.env` existe en la raíz, define `HF_TOKEN` no vacío, y sigue sin trackear (`git ls-files .env` sale vacío); y `git check-ignore -v .env` imprime la regla que lo cubre. Si falta cualquiera de estas condiciones, el barrido **no** arrancó.
- **Dado** `data/2026/detalle/`, **entonces** contiene exactamente 14 archivos `.csv`, cuyos nombres son los `slug` de los 14 modelos del roster, sin sobrantes ni faltantes.
- **Dado** cada CSV, **entonces** tiene exactamente 32 filas, columnas idénticas y en el orden de `COLUMNAS_DETALLE` (17), `idx` de 0 a 31 sin huecos ni repeticiones, un único valor en la columna `modelo` igual al nombre del roster, y todas las latencias estrictamente positivas.
- **Dado** el concatenado de los 14, **entonces** tiene 448 filas y ningún par `(modelo, idx)` duplicado.
- **Dado** cada CSV, **entonces** su columna `modo_prompting` tiene un único valor, y ese valor es `raw_completion` para `LFM2.5-230M` y `LFM2.5-350M`; cualquier discrepancia con la tabla §2.1 del índice quedó **corregida en el índice**, no ocultada.
- **Dado** cada CSV, **entonces** su columna `transformers_version` tiene un único valor, consistente con el `transformers_pin` de ese modelo, y esas versiones efectivas están tabuladas en `docker/README.md`.
- **Dado** el conjunto de exactitudes estrictas por modelo, **entonces** están en `[0, 100]` y no son todas idénticas (todas iguales indicaría que el barrido no varió realmente de modelo).
- **Dado** el barrido, **cuando** se interrumpió y se retomó, **entonces** los modelos ya completos no se recalcularon y el resultado final es indistinguible de una corrida sin interrupciones.
- **Dado** el repositorio tras el commit, **entonces** los archivos de F0 siguen byte-idénticos a `main`, `pytest -q` pasa, y `data/2026/log_barrido.txt` contiene el registro de la corrida.
- **Dado** cualquier fallo de autenticación durante el barrido, **entonces** el proceso se detuvo sin reintentar y sin ejecutar ningún comando de login/auth/configure, sin importar si el modelo involucrado es uno de los dos *gated* o uno de los doce restantes, y quedó reportado como `AUTH-BLOCKER`.
