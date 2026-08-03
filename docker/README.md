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

### Grupo A -- `BASELINE_TRANSFORMERS = "transformers>=4.57.0,<5.0.0"`, resuelve 4.57.6 (8 modelos)

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

### Grupo B -- `TRANSFORMERS_5X = "transformers>=5.0.0"`, resuelve 5.14.1 (4 modelos)

| Modelo | ¿Necesario? | Evidencia (`motivo_pin` de `src/models_2026.py`) |
|---|---|---|
| `LFM2.5-230M` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class TokenizersBackend does not exist or is not currently imported'. PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log). |
| `LFM2.5-350M` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class TokenizersBackend does not exist or is not currently imported'. PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log). |
| `Qwen3.5-0.8B` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError "You can update Transformers with the command 'pip install --upgrade transformers'..." (.claude-scratch/logs/probe.log). |
| `Qwen3.5-2B` | **si** | Necesario: falla bajo transformers 4.57.6 con ValueError "You can update Transformers with the command 'pip install --upgrade transformers'..." (.claude-scratch/logs/probe.log). Su confirmacion positiva bajo 5.14.1 queda **pendiente**: la sonda no llego a correr por el bloqueo de infraestructura de Docker (ver abajo); la evidencia negativa bajo 4.57.6 ya alcanza para fijar el pin. |

**La evidencia clave.** El grupo B resuelve a `transformers 5.14.1`, que es precisamente la version
que rompe a `granite-4.0-350m` (grupo A). Es la prueba mas limpia posible de que ninguna version
mayor unica cubre el roster: 4.57.6 y 5.14.1 son ambas necesarias y mutuamente excluyentes. La
premisa del commit `d6c7581` (que una unica version mayor elimina el confusor) queda refutada por
esto, y el confusor "versiones de libreria divergentes entre modelos" se declara en metodologia y
en Amenazas a la Validez del paper, no se descarta.

## Exclusiones del roster activo

| Modelo | Endpoint que devolvio 403 | Fecha |
|---|---|---|
| `google/gemma-3-270m-it` | `/google/gemma-3-270m-it/resolve/main/config.json` | 2026-08-03 |
| `meta-llama/Llama-3.2-1B-Instruct` | `/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json` | 2026-08-03 |

Los dos repos estan marcados `gated: "manual"`: el token es valido (verificado con `whoami-v2` y
con `/api/models/<id>`, ambos 200), pero el unico endpoint que prueba acceso de DESCARGA es
`/<id>/resolve/<rev>/<archivo>`, y ese devuelve 403 para los dos. Permanecen en `MODELOS_2026`
(`gated=True`, `activo=False`) para que reactivarlos, si la aprobacion llega, sea un cambio de flag
y no una reescritura de codigo.

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

## Uso

    python docker/build_all.py            # construye las 12 imágenes del roster activo
    python docker/run_sweep.py            # corre el barrido completo, de a una
    python docker/run_sweep.py --desde "Qwen3.5-2B"   # retoma tras una interrupción
    python docker/build_all.py --modelo "gemma-3-270m-it"   # falla: ValueError con el motivo_exclusion

El barrido es reanudable en dos niveles (RF3): `src/run_sweep_2026.py` saltea
solo los modelos cuyo CSV de detalle ya está completo y bien formado, y
`--desde` permite además arrancar directamente en el modelo que falló, sin
volver a levantar los contenedores anteriores.

**Bloqueo de infraestructura (2026-08-03).** El almacenamiento del daemon de
Docker quedó en modo solo lectura (`mkdir .../overlay2/...-init: read-only
file system`), así que la reconstrucción real de las 4 imágenes del grupo B
con el pin nuevo queda pendiente hasta que se repare Docker Desktop. No
afecta al código, a los tests ni a esta matriz, que son independientes de
Docker.
