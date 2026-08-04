---
id: 08
title: Ejecución de la etapa 2
depends_on: [07]
files:
  - data/2026/etiquetas_errores.csv
  - data/2026/categorias_comandos.csv
  - data/2026/log_juez.txt
---

## Spec

Ejecutar la etapa 2 con el juez real elegido en `data/2026/juez_seleccionado.json`, produciendo los dos artefactos de datos de **F5**: `etiquetas_errores.csv` (una fila por respuesta incorrecta, **RF8**) y `categorias_comandos.csv` (32 filas, **RF9**).

No se escribe código: se ejecuta `src/judge_2026.py` dentro de la imagen del juez y se validan los invariantes sobre los datos producidos. Se corre en la imagen Docker de ese modelo, no en el host, para que use su `transformers_pin` y su caché ya poblada.

Duración esperada: bastante menor al barrido (una sola carga de modelo, ~200–300 consultas cortas), del orden de 30–90 minutos en CPU.

## Implementation plan

> Exención de TDD completa: es ejecución del comportamiento ya testeado en el subtask 07. La corrección se verifica por invariantes sobre los datos.

### Tarea 1 — Preparación

- [ ] Verificar que existen las entradas y ver quién es el juez:

```bash
python -c "
import json
j = json.load(open('data/2026/juez_seleccionado.json', encoding='utf-8'))
print('Juez:', j['modelo'], '|', j['hf_repo_id'], '|', j['exact_match_pct'], '%')
print('Empatados:', j['empatados'])
"
wc -l data/2026/detalle_2026.csv     # -> 385 (384 + encabezado); menos si hubo exclusiones F12.4
```

- [ ] Calcular cuántas consultas va a hacer, para dimensionar la corrida:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/2026/detalle_2026.csv')
incorrectas = int((~df['match_exact'].astype(bool)).sum())
print(f'consultas de error: {incorrectas} | de categoria: {df[\"idx\"].nunique()}')
print(f'total minimo: {incorrectas + df[\"idx\"].nunique()} (mas reintentos)')
"
```

- [ ] Reconstruir la imagen del juez si hiciera falta:
  `python docker/build_all.py --modelo "<nombre del juez>"`

### Tarea 2 — Ejecución

- [ ] Resolver el juez y su imagen:

```bash
JUEZ=$(python -c "import json;print(json.load(open('data/2026/juez_seleccionado.json',encoding='utf-8'))['modelo'])")
SLUG=$(python -c "import sys;sys.path.insert(0,'src');from models_2026 import slug;print(slug('$JUEZ'))")
```

- [ ] **RF20b — commitear los lotes a medida que salen, no al final.** El juez persiste
  `etiquetas_errores.csv` y `categorias_comandos.csv` por lote (**F12.3**: al terminar cada modelo y
  cada comando, respectivamente), escritos atómicamente en el volumen montado, así que ya son legibles
  desde el host mientras el contenedor sigue corriendo. Motivo registrado: esta sesión ya perdió
  trabajo tres veces (reinicio del backend de Docker, salida del proceso, error 529) — lo que solo
  vive en el working tree es exactamente lo que cuesta una caída. Correr el juez en segundo plano y
  commitear cada vez que cambien los CSV:

```bash
# --cpuset-cpus=0-1 es el mismo valor que usan las 12 corridas del barrido (F11
# amendado): las dos etapas comparten un único envelope de recursos.
docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  "slm-domotica-2026:${SLUG}" \
  python src/judge_2026.py > data/2026/log_juez.txt 2>&1 &
PID_JUEZ=$!

HASH_PREVIO=""
while kill -0 "$PID_JUEZ" 2>/dev/null; do
  sleep 60
  HASH_ACTUAL=$(md5sum data/2026/etiquetas_errores.csv data/2026/categorias_comandos.csv 2>/dev/null)
  if [ -n "$HASH_ACTUAL" ] && [ "$HASH_ACTUAL" != "$HASH_PREVIO" ]; then
    git add data/2026/etiquetas_errores.csv data/2026/categorias_comandos.csv
    git commit -m "data(2026): avance parcial de la etapa 2 (lote persistido por el juez)"
    HASH_PREVIO="$HASH_ACTUAL"
  fi
done
wait "$PID_JUEZ"
```

  Mensaje de commit parcial congelado: `data(2026): avance parcial de la etapa 2 (lote persistido por
  el juez)`. El commit de cierre de la Tarea 4 (`data(2026): etapa 2 ejecutada, ...`) se hace igual al
  final, sobre el estado ya completo — los commits parciales no lo reemplazan.
- [ ] Revisar en el log el conteo de fallos de parseo que imprime el script. Un número alto (> 20 % de las consultas) es señal de que el juez es demasiado débil para la tarea: **no se lo maquilla**, se reporta en el paper y se discute como limitación en §6.

### Tarea 3 — Validación de los artefactos

- [ ] Validar cobertura y vocabulario:

```bash
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from judge_2026 import COLUMNAS_CATEGORIAS, COLUMNAS_ETIQUETAS
from taxonomia_2026 import CATEGORIAS_LINGUISTICAS, ETIQUETAS_ERROR

det = pd.read_csv('data/2026/detalle_2026.csv')
eti = pd.read_csv('data/2026/etiquetas_errores.csv')
cat = pd.read_csv('data/2026/categorias_comandos.csv')

assert list(eti.columns) == COLUMNAS_ETIQUETAS, list(eti.columns)
assert list(cat.columns) == COLUMNAS_CATEGORIAS, list(cat.columns)

incorrectas = det[~det['match_exact'].astype(bool)][['modelo','idx']]
assert len(eti) == len(incorrectas), (len(eti), len(incorrectas))
assert set(map(tuple, eti[['modelo','idx']].values)) == set(map(tuple, incorrectas.values))
assert not eti.duplicated(['modelo','idx']).any()

for v in eti['etiquetas']:
    partes = str(v).split(';')
    assert partes and all(p in ETIQUETAS_ERROR for p in partes), v

assert len(cat) == 32 and cat['idx'].tolist() == list(range(32))
assert cat['categoria'].isin(CATEGORIAS_LINGUISTICAS).all()
print('cobertura y vocabulario OK')
print('fallos de parseo:', int((~eti['juez_parse_ok']).sum()), 'errores,',
      int((~cat['juez_parse_ok']).sum()), 'categorias')
"
```

- [ ] Comparar la distribución de categorías contra la referencia del paper original (8/8/6/4/3/3). **No es una restricción**: si difiere, es un hallazgo a reportar y discutir en §4.3, no algo a forzar.

```bash
python -c "
import pandas as pd
cat = pd.read_csv('data/2026/categorias_comandos.csv')
ref = {'encendido_apagado_simple':8,'ajuste_con_valor_numerico':8,
       'ajuste_relativo_cualitativo':6,'multiples_dispositivos':4,
       'consulta_de_estado':3,'negacion_contexto':3}
obt = cat['categoria'].value_counts().to_dict()
print(f'{\"categoria\":32s} {\"juez\":>5s} {\"paper\":>6s}')
for k in ref:
    print(f'{k:32s} {obt.get(k,0):5d} {ref[k]:6d}')
assert sum(obt.values()) == 32
"
```

- [ ] Verificar el determinismo reejecutando la etapa 2 y comparando hashes (**RNF3**). Se corre en
  **modo default, sin `--reanudar`**: `--reanudar` es opt-in (F12.3) y saltearía todas las filas ya
  presentes, dejando el CSV reescrito sin volver a consultar al juez — el chequeo de byte-identidad
  quedaría vacío en vez de medir determinismo real:

```bash
md5sum data/2026/etiquetas_errores.csv data/2026/categorias_comandos.csv > /tmp/juez_antes.txt
docker run --rm --memory=8g --cpus=2 --cpuset-cpus=0-1 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  "slm-domotica-2026:${SLUG}" python src/judge_2026.py >/dev/null
md5sum -c /tmp/juez_antes.txt && echo "juez deterministico OK"
```

### Tarea 4 — commit

- [ ] `pytest -q`
- [ ] `git add data/2026/etiquetas_errores.csv data/2026/categorias_comandos.csv data/2026/log_juez.txt`
- [ ] `git commit -m "data(2026): etapa 2 ejecutada, etiquetas de error y categorias linguisticas automaticas"`

### Protocolo de fallo

- [ ] Si el juez no carga en su imagen → volver al subtask 04, ajustar el pin, documentar el motivo, rehacer la imagen.
- [ ] Si la tasa de fallos de parseo supera el 20 % → **no** cambiar de juez por cuenta propia: la regla de selección es un contrato congelado (RF7). Registrar el número, seguir adelante, y asegurarse de que los subtasks 14/15 lo reporten en §4.4 y lo discutan en §6.
- [ ] Si aparece cualquier 401 / 403 → **PARAR**, no reintentar, no ejecutar ningún comando de login.

## Verify

```bash
# 1. Los dos artefactos existen con el esquema congelado
python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd
from judge_2026 import COLUMNAS_ETIQUETAS, COLUMNAS_CATEGORIAS
assert list(pd.read_csv('data/2026/etiquetas_errores.csv').columns) == COLUMNAS_ETIQUETAS
assert list(pd.read_csv('data/2026/categorias_comandos.csv').columns) == COLUMNAS_CATEGORIAS
print('esquemas OK')
"

# 2. Cobertura exacta de las incorrectas y de los 32 comandos
python -c "
import pandas as pd
det = pd.read_csv('data/2026/detalle_2026.csv')
eti = pd.read_csv('data/2026/etiquetas_errores.csv')
cat = pd.read_csv('data/2026/categorias_comandos.csv')
inc = set(map(tuple, det[~det['match_exact'].astype(bool)][['modelo','idx']].values))
assert set(map(tuple, eti[['modelo','idx']].values)) == inc
assert sorted(cat['idx']) == list(range(32))
print(f'{len(inc)} incorrectas etiquetadas, 32 comandos categorizados')
"

# 3. Las etiquetas de equivalencia existen: si no hay NINGUNA, la métrica laxa
#    sería idéntica a la estricta y el resultado central del paper se cae
python -c "
import pandas as pd
eti = pd.read_csv('data/2026/etiquetas_errores.csv')
equiv = eti['etiquetas'].str.contains('sin_error_semantico|uso_de_sinonimos').sum()
print(f'filas rescatadas por la metrica laxa: {equiv} / {len(eti)}')
"

# 4. Determinismo verificado (ver Tarea 3)

# 5. Suite verde y dataset intacto
pytest -q
git diff --exit-code main -- data/dataset_comandos_domotica.csv \
  data/resultados_experimento_detalle.csv && echo "F0 intacto"
```

## Acceptance criteria

- **Dado** `data/2026/etiquetas_errores.csv`, **entonces** tiene exactamente las columnas `modelo,idx,etiquetas,juez_raw,juez_parse_ok`, una fila por cada `(modelo, idx)` con `match_exact == False` en `detalle_2026.csv`, sin duplicados y sin faltantes.
- **Dado** cualquier valor de la columna `etiquetas`, **entonces** al separarlo por `;` todas sus partes pertenecen a `ETIQUETAS_ERROR` y hay al menos una.
- **Dado** `data/2026/categorias_comandos.csv`, **entonces** tiene exactamente 32 filas con `idx` de 0 a 31 sin repetir, columnas `idx,comando,categoria,juez_raw,juez_parse_ok`, y toda `categoria` pertenece a `CATEGORIAS_LINGUISTICAS`.
- **Dado** que el juez falló al parsear en algún caso, **entonces** esas filas tienen `juez_parse_ok == False` con una etiqueta de respaldo válida, y el conteo total quedó registrado en `data/2026/log_juez.txt` para reportarlo en el paper.
- **Dado** que se reejecuta `src/judge_2026.py` sobre el mismo `detalle_2026.csv`, **entonces** ambos CSV quedan byte-idénticos (determinismo por decodificación greedy, RNF3).
- **Dado** la distribución de categorías, **entonces** suma 32 y fue comparada contra la referencia 8/8/6/4/3/3 del paper original; cualquier diferencia quedó registrada para discutirla, no corregida a mano.
- **Dado** el juez, **entonces** es exactamente el modelo indicado en `juez_seleccionado.json`, ejecutado dentro de su propia imagen Docker con su `transformers_pin`; no se sustituyó por otro modelo aunque su tasa de parseo fuera baja.
- **Dado** cualquier invocación de `docker run` de este subtask (ejecución de la Tarea 2 y re-corrida de determinismo de la Tarea 3), **entonces** lleva `--cpuset-cpus=0-1` junto con `--cpus=2 --memory=8g`, el mismo valor que usan las 12 corridas del barrido (F11 amendado).
- **Dado** RF20b, **entonces** hubo al menos un commit parcial con el mensaje `data(2026): avance parcial de la etapa 2 (lote persistido por el juez)` antes del commit de cierre de la Tarea 4, evidenciando que los CSV se versionaron a medida que el juez los persistía por lote (F12.3) y no solo al final.
- **Dado** la re-corrida de determinismo de la Tarea 3, **entonces** se ejecuta en modo default (sin `--reanudar`); usar `--reanudar` ahí volvería vacío el chequeo de byte-identidad, porque saltearía todas las filas ya presentes en vez de volver a consultar al juez.
- **Dado** el commit, **entonces** `pytest -q` pasa, los archivos de F0 siguen intactos, y no se ejecutó ningún comando de autenticación en ningún momento.
