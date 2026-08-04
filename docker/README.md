# Aislamiento por modelo y matriz de versiones

Cada modelo del roster 2026 corre en su propia imagen, construida desde
`Dockerfile.modelo` con su propio `--build-arg TRANSFORMERS_PIN`. Las imágenes
comparten un único volumen de caché de pesos (`.hf_cache`): se aíslan los
árboles de dependencias, no las descargas.

Baseline del proyecto: `transformers>=4.57.0,<5.0.0` (grupo A). Ver la nota de
versiones de F3 en el índice (`TODO.md` §4) para la narrativa completa; esta
matriz es la fuente de datos, esa nota es la fuente de la interpretación.

## Matriz de versiones

Dos grupos de version, no doce pines distintos. Evidencia por modelo, nunca inferida por familia
(ver `.claude-scratch/logs/probe.log`, `.claude-scratch/logs/probe5x.log`,
`.claude-scratch/logs/sweep2.log` -- sondeos de compatibilidad, nunca datos de latencia).

### Grupo A -- `BASELINE_TRANSFORMERS = "transformers>=4.57.0,<5.0.0"`, resuelve 4.57.6 (10 modelos: 9 del roster activo + 1 baseline-completion excluida)

| Modelo | ¿Necesario? | Evidencia |
|---|---|---|
| `granite-4.0-350m` | **si** | Falla bajo 5.14.1: `ValueError: has_previous_state can only be called on LinearAttention layers, and the current Cache seem to only contain Attention layers` (regresion de transformers 5.x en el cache hibrido de Granite 4). `PROBE_OK` en 4.57.6. |
| `granite-4.0-h-350m` | heredado | `PROBE_OK` en 4.57.6; 5.x no probado. |
| `granite-4.0-1b` | heredado | idem. |
| `granite-4.0-h-1b` | heredado | idem. |
| `SmolLM2-360M-Instruct` | heredado | idem. |
| `SmolLM2-1.7B-Instruct` | heredado | idem. |
| `LFM2.5-1.2B-Instruct` | heredado | idem. |
| `OLMo-2-0425-1B-Instruct` | heredado | idem. |
| `Qwen2.5-1.5B-Instruct` | heredado | `PROBE_OK` en 4.57.6 (`PROBE_OK\|Qwen2.5-1.5B-Instruct\|4.57.6\|chat_template`, `.claude-scratch/logs/probe_qwen25_15b_groupA.log`); 5.x no probado. Grupo determinado por sonda propia, nunca por inferencia de familia: Qwen3.5 resuelve a grupo B. |
| `Qwen2.5-0.5B-Instruct` | heredado | **Baseline-completion (Delta 05, 2026-08-04), excluida del roster activo, no gated.** `PROBE_OK` en 4.57.6 (`PROBE_OK\|Qwen2.5-0.5B-Instruct\|4.57.6\|chat_template`, `.claude-scratch/logs/probe_qwen25_05b_groupA.log`), sondeada dentro de la misma imagen de grupo A ya construida (`slm-domotica-2026:granite-4-0-350m`); 5.x no probado. Es el 4to modelo del paper original (`baseline_original=True`): completa el baseline re-medido bajo el harness/prompt 2026 sin entrar al roster de 12. |

### Grupo B -- `TRANSFORMERS_5X = "transformers>=5.0.0"`, resuelve 5.14.1 (3 modelos activos + 1 registrado y excluido)

| Modelo | ¿Necesario? | Evidencia (`motivo_pin` de `src/models_2026.py`) |
|---|---|---|
| `LFM2.5-230M` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class TokenizersBackend does not exist or is not currently imported'. PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log). |
| `LFM2.5-350M` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class TokenizersBackend does not exist or is not currently imported'. PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log). |
| `Qwen3.5-0.8B` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError "You can update Transformers with the command 'pip install --upgrade transformers'..." (.claude-scratch/logs/probe.log). |
| `Qwen3.5-2B` | **si** (registrado, **excluido del roster activo**) | Necesario: falla bajo transformers 4.57.6 con ValueError "You can update Transformers with the command 'pip install --upgrade transformers'..." (`.claude-scratch/logs/probe.log`). Su confirmacion positiva bajo 5.14.1 **nunca se obtuvo**: la sonda no llego a correr por el bloqueo de infraestructura de Docker. **Compatibilidad no verificada, no incompatibilidad demostrada.** Fuera del roster activo por decision del usuario (2026-08-04). |

**La evidencia clave.** El grupo B resuelve a `transformers 5.14.1`, que es precisamente la version
que rompe a `granite-4.0-350m` (grupo A). Es la prueba mas limpia posible de que ninguna version
mayor unica cubre el roster: 4.57.6 y 5.14.1 son ambas necesarias y mutuamente excluyentes. La
premisa del commit `d6c7581` (que una unica version mayor elimina el confusor) queda refutada por
esto, y el confusor "versiones de libreria divergentes entre modelos" se declara en metodologia y
en Amenazas a la Validez del paper, no se descarta.

## Exclusiones del roster activo

| Modelo | Causa | Detalle | Fecha |
|---|---|---|---|
| `google/gemma-3-270m-it` | acceso de descarga no otorgado | 403 en `/google/gemma-3-270m-it/resolve/main/config.json` con `$HF_TOKEN` valido (repo `gated: "manual"`) | 2026-08-03 |
| `meta-llama/Llama-3.2-1B-Instruct` | acceso de descarga no otorgado | 403 en `/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json` con `$HF_TOKEN` valido (repo `gated: "manual"`) | 2026-08-03 |
| `Qwen/Qwen3.5-2B` | compatibilidad de version **no verificada** | Falla bajo 4.57.6 (evidencia propia). Su sonda bajo 5.14.1 nunca corrio: el almacenamiento del daemon de Docker quedo en solo lectura. **No** hay evidencia de que falle bajo 5.14.1. Decision del usuario: excluirlo en lugar de perseguir la confirmacion. **No es gated**: su invocacion no lleva `--env-file`. | 2026-08-04 |
| `Qwen/Qwen2.5-0.5B-Instruct` | **baseline-completion**, no es un modelo nuevo del roster 2026 | Cuarto modelo de `roster_baseline_original()` (Delta 05): completa la medicion de los 4 modelos del paper original bajo el harness/prompt 2026, motivada porque un control con el prompt original NO reprodujo la Tabla 2 publicada. `activo=False` sin condicion -- el roster activo de 12 no se revisita. `PROBE_OK` en 4.57.6, **no gated** (su invocacion tampoco lleva `--env-file`); a diferencia de los otros tres excluidos, esta exclusion es de alcance del roster, no de acceso ni de version. Se construye igual, explicitamente, via `docker/build_all.py --modelo "Qwen2.5-0.5B-Instruct"` y corre bajo `docker/run_baseline.py`, no bajo `docker/run_sweep.py`. | 2026-08-04 |

Los dos primeros repos estan marcados `gated: "manual"`: el token es valido (verificado con
`whoami-v2` y con `/api/models/<id>`, ambos 200), pero el unico endpoint que prueba acceso de
DESCARGA es `/<id>/resolve/<rev>/<archivo>`, y ese devuelve 403 para los dos. Permanecen en
`MODELOS_2026` (`gated=True`, `activo=False`) para que reactivarlos, si la aprobacion llega, sea un
cambio de flag y no una reescritura de codigo. `Qwen3.5-2B` es un caso distinto: no es gated, y su
exclusion es por compatibilidad de version no verificada bajo 5.14.1, no por acceso denegado.
`Qwen2.5-0.5B-Instruct` es un tercer caso distinto de los dos anteriores: tampoco es gated y su
version SI esta verificada (`PROBE_OK` en 4.57.6); su exclusion es de alcance de roster
(baseline-completion, Delta 05), no de acceso ni de version.

## Credenciales de los modelos gated

El roster activo tiene **cero** modelos gated; esta sección describe la plomería que se conserva por
si se reactiva alguno de los dos excluidos: `google/gemma-3-270m-it` y
`meta-llama/Llama-3.2-1B-Instruct`. El token
de Hugging Face vive **solo** en `.env` en la raíz del repo (ignorado por
git) y se inyecta **únicamente en tiempo de ejecución**, vía
`docker run --env-file .env`, y **solo** a esas dos imágenes si se reactivan; las otras 12
corren sin credenciales.

Prohibido: declarar `HF_TOKEN` como `ARG` o como `ENV` en `Dockerfile.modelo`,
pasar el token como `--build-arg`, copiar `.env` al contexto de build o a la
imagen, escribirlo en una capa, o invocar el login interactivo de la CLI de
Hugging Face (ni su equivalente en `huggingface_hub`). La única forma de
consumirlo es leer `os.environ["HF_TOKEN"]` en runtime (ver
`src/run_sweep_2026.py`, subtask 03).

Estas prohibiciones están testeadas, no solo escritas: ver
`tests/test_docker_matriz.py` (nada bajo `docker/` declara el token en una
capa ni hace login) y `tests/test_credenciales_gated.py` (el barrido aborta
temprano si falta el token).

Si la lista a correr incluye algún modelo gated y `.env` no existe o
`HF_TOKEN` está ausente o vacío, `docker/run_sweep.py` aborta con un
`ValueError` en español antes de construir o correr nada; el mensaje nunca
imprime el valor del token.

Formato de `.env` en la raíz del repo (nunca se versiona):

    HF_TOKEN=<tu token de Hugging Face>

## Envelope de recursos (RNF1, RNF6)

Toda invocacion de `docker run` de este proyecto -- las 12 del barrido y la del
juez -- usa:

    --memory=8g --cpus=2 --cpuset-cpus=0-1

`--cpus=2` es una **cuota** del planificador CFS: acota el tiempo de CPU pero
permite migracion entre nucleos y no reserva nada. `--memory=8g` es un **techo**,
no una reserva (Docker solo ofrece `--memory-reservation`, que es blanda).
`--cpuset-cpus` **si** fija los nucleos y elimina tanto la competencia por tiempo
de CPU como la variabilidad por migracion. Lo que ningun flag puede hacer es
particionar la cache L3 ni el bus de memoria, que es donde la inferencia de LLM
en CPU esta realmente limitada: por eso el barrido es **secuencial**, un modelo a
la vez. Correr 4 modelos en paralelo habria inflado los tiempos por comando de
forma invisible en la tabla final. Costo de la decision: ~7 h de reloj en lugar
de ~2 h. Exactitud, taxonomia y equivalencia semantica no dependen de esto
(temperatura 0, deterministas); solo la latencia lo hacia.

Lo que hace comparable la latencia no es *cual* par de nucleos se use, sino que
sea **el mismo par en las 12 corridas y en la del juez**. Para cambiarlo en otro
host: `python docker/run_sweep.py --cpuset "2-3"`.

## Uso

    python docker/build_all.py            # construye las 12 imagenes del roster activo
    python docker/run_sweep.py            # corre el barrido completo, de a una, --cpuset-cpus=0-1
    python docker/run_sweep.py --desde "Qwen3.5-0.8B"   # retoma tras una interrupcion
    python docker/run_sweep.py --cpuset "2-3"           # otro par de nucleos, el mismo para las 12
    python docker/build_all.py --modelo "gemma-3-270m-it"   # falla: ValueError con el motivo_exclusion
    python docker/run_sweep.py --desde "Qwen3.5-2B"         # falla: excluido del roster activo

El barrido es reanudable en dos niveles (RF3): `src/run_sweep_2026.py` saltea
solo los modelos cuyo CSV de detalle ya está completo y bien formado, y
`--desde` permite además arrancar directamente en el modelo que falló, sin
volver a levantar los contenedores anteriores.

**Continuación ante fallo (F12.1, RF20c).** El barrido **no se detiene** por un
modelo que falla: registra el fallo -- con su error textual -- en
`data/2026/fallos_barrido.json` (lista, siempre existe salvo en `--dry-run`) y
sigue con el siguiente modelo, porque el usuario no está disponible para
desbloquearlo y los modelos restantes son datos que se perderían si el barrido
cortara. Un `401`/`403`, en cambio, **sí** corta el proceso de inmediato
(AUTH-STOP) y no se reintenta: la continuación ante fallo aplica a fallos
técnicos del modelo, nunca a fallos de autenticación.

**Estado de infraestructura (actualizado al 2026-08-04).** El bloqueo de
almacenamiento en modo solo lectura del daemon de Docker documentado el
2026-08-03 quedó resuelto: la sonda de `Qwen2.5-1.5B-Instruct` corrió dentro de
la imagen ya construida `slm-domotica-2026:granite-4-0-350m`, así que el daemon
volvió a estado de escritura. Lo que queda pendiente son las **3**
reconstrucciones del grupo B (`LFM2.5-230M`, `LFM2.5-350M`, `Qwen3.5-0.8B`) y la
**1** construcción nueva del grupo A (`Qwen2.5-1.5B-Instruct`); las 8 imágenes
restantes del grupo A no se reconstruyen (su pin no cambió).
