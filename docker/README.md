# Aislamiento por modelo y matriz de versiones

Cada modelo del roster 2026 corre en su propia imagen, construida desde
`Dockerfile.modelo` con su propio `--build-arg TRANSFORMERS_PIN`. Las imágenes
comparten un único volumen de caché de pesos (`.hf_cache`): se aíslan los
árboles de dependencias, no las descargas.

Baseline del proyecto: `transformers>=4.57.0`.

## Matriz de versiones

| Modelo | `transformers_pin` | `trust_remote_code` | ¿Divergente? | Motivo | ¿Gated? |
|---|---|---|---|---|---|
| LFM2.5-230M | `transformers>=4.57.0` | no | no | — | no |
| LFM2.5-350M | `transformers>=4.57.0` | no | no | — | no |
| granite-4.0-350m | `transformers>=4.57.0` | no | no | — | no |
| granite-4.0-h-350m | `transformers>=4.57.0` | no | no | — | no |
| Qwen3.5-0.8B | `transformers>=4.57.0` | no | no | — | no |
| SmolLM2-360M-Instruct | `transformers>=4.57.0` | no | no | — | no |
| gemma-3-270m-it | `transformers>=4.57.0` | no | no | — | **sí** |
| LFM2.5-1.2B-Instruct | `transformers>=4.57.0` | no | no | — | no |
| granite-4.0-1b | `transformers>=4.57.0` | no | no | — | no |
| granite-4.0-h-1b | `transformers>=4.57.0` | no | no | — | no |
| Qwen3.5-2B | `transformers>=4.57.0` | no | no | — | no |
| OLMo-2-0425-1B-Instruct | `transformers>=4.57.0` | no | no | — | no |
| SmolLM2-1.7B-Instruct | `transformers>=4.57.0` | no | no | — | no |
| Llama-3.2-1B-Instruct | `transformers>=4.57.0` | no | no | — | **sí** |

Los modelos marcados como no divergentes heredan el baseline; su pin no es una
decisión, es la ausencia de una. Solo las filas marcadas como divergentes
representan una restricción real descubierta empíricamente, y son las que se
reportan en la Tabla 5 y en la Sección 6 (Amenazas a la validez) del paper:
correr distintos modelos con distintas versiones de la librería de inferencia
es un confusor para la comparación de latencia.

Al cierre del subtask 04 la matriz **no tiene divergencias**: los 14 modelos
heredan el baseline y ninguno necesita `trust_remote_code`. Es el mejor caso
posible para la validez interna del estudio, porque elimina ese confusor. Si
la ejecución real del barrido (subtask 05) descubriera que algún modelo no
carga con el baseline, la corrección va en los campos `transformers_pin` /
`trust_remote_code` / `motivo_pin` de `src/models_2026.py` y en una fila de
esta tabla, en el mismo commit — el test
`test_todo_pin_divergente_esta_justificado_en_el_readme` no deja que una cosa
avance sin la otra.

> Nota informativa: Gemma 3 requiere `transformers >= 4.50.0`; cubierto por el
> baseline `transformers>=4.57.0`. No es un pin divergente — `gemma-3-270m-it`
> hereda el baseline igual que los demás.

## Credenciales de los modelos gated

Dos modelos del roster son *gated*: `google/gemma-3-270m-it` y
`meta-llama/Llama-3.2-1B-Instruct` (columna "¿Gated?" de la tabla). El token
de Hugging Face vive **solo** en `.env` en la raíz del repo (ignorado por
git) y se inyecta **únicamente en tiempo de ejecución**, vía
`docker run --env-file .env`, y **solo** a esas dos imágenes; las otras 12
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

    python docker/build_all.py            # construye las 14 imágenes
    python docker/run_sweep.py            # corre el barrido completo, de a una
    python docker/run_sweep.py --desde "Qwen3.5-2B"   # retoma tras una interrupción

El barrido es reanudable en dos niveles (RF3): `src/run_sweep_2026.py` saltea
solo los modelos cuyo CSV de detalle ya está completo y bien formado, y
`--desde` permite además arrancar directamente en el modelo que falló, sin
volver a levantar los contenedores anteriores.
