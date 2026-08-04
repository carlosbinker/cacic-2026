"""Registro de los 14 modelos evaluados en el experimento 2026 (F3 del índice).

Puro dato + validación: sin dependencias de `torch`/`transformers`, para que se
pueda importar y testear en cualquier contexto, incluidos los contenedores
mínimos del subtask 04. Declara el flag `gated` únicamente; nunca lee ni
imprime el valor de ningún token de HuggingFace.

Delta 2026-08-03 (corrige hacia adelante a `d6c7581`, sin editarlo). Dos cosas
que Phase 3 refutó empíricamente:

1. No existe una única versión mayor de `transformers` que cubra el roster.
   `granite-4.0-350m` falla bajo 5.14.1; `LFM2.5-230M`, `LFM2.5-350M`,
   `Qwen3.5-0.8B` y `Qwen3.5-2B` fallan bajo 4.57.6. 4.57.x y 5.x son ambas
   NECESARIAS y mutuamente excluyentes sobre el roster: el pin vuelve a ser
   por modelo, agrupado en dos grupos (`BASELINE_TRANSFORMERS`,
   `TRANSFORMERS_5X`). El confusor "versiones de librería divergentes" se
   declara, no se elimina. Matriz completa y evidencia por modelo en
   `docker/README.md`.
2. El roster ACTIVO del barrido pasa a ser 12, no 14: los dos modelos gated
   (`google/gemma-3-270m-it`, `meta-llama/Llama-3.2-1B-Instruct`) siguen en el
   REGISTRO pero quedan `activo=False` porque el acceso de descarga no fue
   otorgado (403 en `/<id>/resolve/main/config.json`, aprobación manual
   pendiente al 2026-08-03). Reactivar uno es `activo=True` + vaciar
   `motivo_exclusion`, nunca una reescritura de código.

Delta 2026-08-04 (corrige hacia adelante a `864c694`, sin editarlo). El
REGISTRO pasa de 14 a 15 filas, a conteo de roster activo constante (12):
3. `Qwen3.5-2B` sale del roster activo sin condición y queda `activo=False`
   por compatibilidad NO VERIFICADA bajo transformers 5.14.1 (no incompatibilidad
   demostrada: su sonda bajo 5.14.1 nunca corrió por un bloqueo de infraestructura
   de Docker, no por evidencia sobre el modelo); conserva su `transformers_pin`
   del grupo B y su `motivo_pin`, porque ese pin sigue siendo necesario por su
   evidencia negativa propia bajo 4.57.6. `Qwen/Qwen2.5-1.5B-Instruct` entra al
   roster activo, grupo A, por sonda propia
   (`PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template`), con pin HEREDADO
   (`motivo_pin == ""`: la sonda bajo 5.14.1 no se corrió, así que no hay
   evidencia de necesidad). El ID `Qwen/Qwen2.5-1.7B-Instruct` que se llegó a
   mencionar NO EXISTE (HTTP 404) y no debe reintroducirse nunca; el ID
   correcto es `Qwen/Qwen2.5-1.5B-Instruct`. El reingreso de este modelo
   sostiene la línea de continuidad con la Tabla 2 publicada (RF19): los
   modelos presentes en los dos estudios pasan de 2 a 3.

Delta 05 (2026-08-04). Un control re-corrió los tres anclajes de continuidad
bajo el prompt ORIGINAL (`SYSTEM_PROMPT_PAPER`) y NO reprodujo la Tabla 2
publicada (Qwen2.5-1.5B-Instruct 50.0% publicado vs 56.2% en control;
SmolLM2-1.7B-Instruct 59.4% vs 50.0%; SmolLM2-360M-Instruct 18.8% vs 0.0%).
La causa, con evidencia: `requirements.txt` legacy fija `transformers>=4.46.0`
SIN cota superior y `data/resultados_experimento_detalle.csv` no tiene columna
`transformers_version` -- el entorno de la corrida original nunca quedó
fijado ni registrado, así que es irrecuperable. Es irreproducibilidad del
trabajo original, no un defecto de `tests/test_metricas.py` (que sigue verde
y recalcula los porcentajes publicados desde los outputs crudos publicados:
la ruta de scoring es fiel).

Consecuencia: las cifras publicadas dejan de ser baseline validado y pasan a
referencia histórica no reproducida. El paper compara en cambio contra un
baseline RE-MEDIDO e internamente consistente: los 4 modelos del paper
original, medidos bajo el harness/prompt 2026, misma máquina, temperatura 0.
4. `Qwen/Qwen2.5-0.5B-Instruct` se agrega al REGISTRO (15 → 16) para
   completar ese baseline: sonda propia confirma grupo A
   (`PROBE_OK|Qwen2.5-0.5B-Instruct|4.57.6|chat_template`, ver
   `.claude-scratch/logs/probe_qwen25_05b_groupA.log`), pin HEREDADO (no se
   probó 5.14.1, mismo estándar que `Qwen2.5-1.5B-Instruct`). Es
   `baseline_original=True` pero `activo=False`: el roster activo de 12 NO
   se revisita (decisión del usuario). Los otros 3 modelos de
   `baseline_original` (`SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`,
   `Qwen2.5-1.5B-Instruct`) ya son parte del roster activo; `activo` y
   `baseline_original` son ejes ortogonales, igual que `activo` y
   `motivo_pin` (F3 (g)). Ver `roster_baseline_original()`.
"""

import re
from dataclasses import dataclass
from typing import Literal

Tier = Literal["sub-1B", "1-2B"]

# Grupo A (re-congelado 2026-08-02, alcance revisado 2026-08-03): cota superior
# <5.0.0 porque transformers 5.14.1 rompe a ibm-granite/granite-4.0-350m en el
# primer generate() ("has_previous_state can only be called on
# LinearAttention layers..."), regresion de transformers 5.x en el manejo de
# cache hibrida/linear-attention para la arquitectura Granite 4. Resuelve a
# transformers 4.57.6. Necesario solo para granite-4.0-350m; los otros 7
# modelos del grupo A lo heredan (corren bajo 4.57.6, no probados bajo 5.x).
BASELINE_TRANSFORMERS: str = "transformers>=4.57.0,<5.0.0"

# Grupo B (2026-08-03): 4 modelos (LFM2.5-230M/350M, Qwen3.5-0.8B/2B) exigen
# transformers >= 5.0.0; ver motivo_pin de cada uno. Resuelve, en esta corrida,
# a transformers 5.14.1 -- que es precisamente la version que rompe a
# granite-4.0-350m (grupo A). Los dos grupos son necesarios y mutuamente
# excluyentes sobre el roster: no existe una unica version mayor que sirva
# para las 12 filas activas. La premisa de d6c7581 queda refutada.
TRANSFORMERS_5X: str = "transformers>=5.0.0"

# Textos de motivo (constantes de módulo: no se repiten strings largos inline
# y `docker/README.md` cita el mismo texto con `[:N]`).
_MOTIVO_PIN_LFM25_BASE = (
    "Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class "
    "TokenizersBackend does not exist or is not currently imported'. "
    "PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log)."
)
_MOTIVO_PIN_QWEN35 = (
    "Necesario: falla bajo transformers 4.57.6 con ValueError \"You can update "
    "Transformers with the command 'pip install --upgrade transformers'...\" "
    "(.claude-scratch/logs/probe.log)."
)
_MOTIVO_PIN_GRANITE_350M = (
    "Necesario: falla bajo transformers 5.14.1 con ValueError 'has_previous_state "
    "can only be called on LinearAttention layers, and the current Cache seem to "
    "only contain Attention layers' (regresion de transformers 5.x en el cache "
    "hibrido de Granite 4; .claude-scratch/logs/sweep2.log). PROBE_OK en 4.57.6 "
    "(.claude-scratch/logs/probe.log)."
)
_MOTIVO_EXCLUSION_GATED = (
    "Acceso de descarga no otorgado: 403 en /{repo}/resolve/main/config.json con "
    "$HF_TOKEN valido (repo gated:manual, aprobacion pendiente al 2026-08-03). "
    "Reactivar es poner activo=True y vaciar este campo, sin reescribir codigo."
)
_MOTIVO_EXCLUSION_BASELINE_ORIGINAL = (
    "No es un modelo nuevo del roster 2026: es el cuarto y ultimo modelo del paper original "
    "que faltaba medir bajo el harness/prompt 2026 para completar un baseline internamente "
    "consistente de los 4 modelos originales (Delta 05, 2026-08-04, motivado por el control que "
    "NO reprodujo la Tabla 2 publicada). Los otros 3 modelos del baseline original "
    "(SmolLM2-360M-Instruct, SmolLM2-1.7B-Instruct, Qwen2.5-1.5B-Instruct) ya estan en el roster "
    "activo. Este NO entra al roster activo de 12: esa decision del usuario no se revisita. Ver "
    "roster_baseline_original() y data/2026/baseline_original/."
)
_MOTIVO_EXCLUSION_QWEN35_2B = (
    "Fuera del roster activo por decision del usuario (2026-08-04): su pin del grupo B es "
    "necesario (falla bajo 4.57.6 con ValueError que pide 'pip install --upgrade "
    "transformers'), pero su confirmacion positiva bajo 5.14.1 NUNCA SE OBTUVO -- la sonda no "
    "llego a correr porque el almacenamiento del daemon de Docker quedo en modo solo lectura a "
    "mitad de ronda (bloqueo de infraestructura, NO evidencia sobre el modelo). Compatibilidad "
    "NO VERIFICADA, no incompatibilidad demostrada: no hay evidencia de que falle bajo 5.14.1. "
    "El usuario eligio excluirlo en lugar de perseguir esa confirmacion. Reactivarlo es poner "
    "activo=True, vaciar este campo y obtener la sonda positiva."
)


@dataclass(frozen=True)
class ModeloEvaluado2026:
    """Un modelo del roster 2026, con su plomería de reproducibilidad."""

    nombre: str  # clave única; valor de la columna `modelo` en todos los CSV
    hf_repo_id: str
    params_b: float
    tier: Tier
    transformers_pin: str  # BASELINE_TRANSFORMERS o TRANSFORMERS_5X
    trust_remote_code: bool
    gated: bool  # requiere licencia aceptada + $HF_TOKEN para descargarse
    motivo_pin: str  # "" si el pin es heredado; el porqué si el pin es NECESARIO
    activo: bool  # True <=> forma parte del roster activo del barrido
    motivo_exclusion: str  # "" si activo; el porqué si no
    # Delta 05: True <=> uno de los 4 modelos del paper original (Tabla 2 publicada).
    # Eje ORTOGONAL a `activo`: 3 de los 4 ya están en el roster activo; el cuarto
    # (Qwen2.5-0.5B-Instruct) es baseline_original=True con activo=False. Default
    # False para no tocar ninguna de las filas que no son del paper original.
    baseline_original: bool = False


MODELOS_2026: list[ModeloEvaluado2026] = [
    ModeloEvaluado2026(
        nombre="LFM2.5-230M",
        hf_repo_id="LiquidAI/LFM2.5-230M",
        params_b=0.23,
        tier="sub-1B",
        transformers_pin=TRANSFORMERS_5X,
        trust_remote_code=False,
        gated=False,
        motivo_pin=_MOTIVO_PIN_LFM25_BASE,
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="LFM2.5-350M",
        hf_repo_id="LiquidAI/LFM2.5-350M",
        params_b=0.35,
        tier="sub-1B",
        transformers_pin=TRANSFORMERS_5X,
        trust_remote_code=False,
        gated=False,
        motivo_pin=_MOTIVO_PIN_LFM25_BASE,
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="granite-4.0-350m",
        hf_repo_id="ibm-granite/granite-4.0-350m",
        params_b=0.35,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin=_MOTIVO_PIN_GRANITE_350M,
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="granite-4.0-h-350m",
        hf_repo_id="ibm-granite/granite-4.0-h-350m",
        params_b=0.34,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="Qwen3.5-0.8B",
        hf_repo_id="Qwen/Qwen3.5-0.8B",
        params_b=0.8,
        tier="sub-1B",
        transformers_pin=TRANSFORMERS_5X,
        trust_remote_code=False,
        gated=False,
        motivo_pin=_MOTIVO_PIN_QWEN35,
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="SmolLM2-360M-Instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        params_b=0.36,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
        baseline_original=True,  # anclaje de continuidad RF19, uno de los 4 del paper original
    ),
    ModeloEvaluado2026(
        nombre="gemma-3-270m-it",
        hf_repo_id="google/gemma-3-270m-it",
        params_b=0.27,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=True,
        motivo_pin="",
        activo=False,
        motivo_exclusion=_MOTIVO_EXCLUSION_GATED.format(repo="google/gemma-3-270m-it"),
    ),
    ModeloEvaluado2026(
        nombre="LFM2.5-1.2B-Instruct",
        hf_repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        params_b=1.2,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="granite-4.0-1b",
        hf_repo_id="ibm-granite/granite-4.0-1b",
        params_b=1.6,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="granite-4.0-h-1b",
        hf_repo_id="ibm-granite/granite-4.0-h-1b",
        params_b=1.5,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="Qwen3.5-2B",
        hf_repo_id="Qwen/Qwen3.5-2B",
        params_b=2.0,
        tier="1-2B",
        transformers_pin=TRANSFORMERS_5X,
        trust_remote_code=False,
        gated=False,
        motivo_pin=_MOTIVO_PIN_QWEN35,
        activo=False,
        motivo_exclusion=_MOTIVO_EXCLUSION_QWEN35_2B,
    ),
    ModeloEvaluado2026(
        nombre="OLMo-2-0425-1B-Instruct",
        hf_repo_id="allenai/OLMo-2-0425-1B-Instruct",
        params_b=1.0,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
    ),
    ModeloEvaluado2026(
        nombre="SmolLM2-1.7B-Instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        params_b=1.71,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
        baseline_original=True,  # anclaje de continuidad RF19, uno de los 4 del paper original
    ),
    ModeloEvaluado2026(
        nombre="Llama-3.2-1B-Instruct",
        hf_repo_id="meta-llama/Llama-3.2-1B-Instruct",
        params_b=1.0,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=True,
        motivo_pin="",
        activo=False,
        motivo_exclusion=_MOTIVO_EXCLUSION_GATED.format(repo="meta-llama/Llama-3.2-1B-Instruct"),
    ),
    ModeloEvaluado2026(
        nombre="Qwen2.5-1.5B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.54,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        # Pin HEREDADO: la sonda del 2026-08-04 confirma que 4.57.6 funciona
        # (PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template, ver
        # .claude-scratch/logs/probe_qwen25_15b_groupA.log), pero NO se probo
        # 5.14.1, asi que no hay evidencia de necesidad. Grupo determinado por
        # sonda propia, nunca por inferencia de familia: Qwen3.5 resuelve a
        # grupo B, o sea que "familia Qwen" no predice version.
        motivo_pin="",
        activo=True,
        motivo_exclusion="",
        baseline_original=True,  # anclaje de continuidad RF19, uno de los 4 del paper original
    ),
    ModeloEvaluado2026(
        nombre="Qwen2.5-0.5B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.494,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        # Pin HEREDADO: la sonda del 2026-08-04 confirma que 4.57.6 funciona
        # (PROBE_OK|Qwen2.5-0.5B-Instruct|4.57.6|chat_template, ver
        # .claude-scratch/logs/probe_qwen25_05b_groupA.log), corrida dentro de la
        # misma imagen ya construida del grupo A (slm-domotica-2026:granite-4-0-350m)
        # que se usó para sondear a Qwen2.5-1.5B-Instruct. NO se probó 5.14.1, así
        # que no hay evidencia de necesidad: mismo estándar que el resto del grupo A.
        motivo_pin="",
        activo=False,  # baseline-completion: no entra al roster activo de 12 (Delta 05)
        motivo_exclusion=_MOTIVO_EXCLUSION_BASELINE_ORIGINAL,
        baseline_original=True,  # el 4to modelo del paper original; completa el baseline
    ),
]


def slug(nombre: str) -> str:
    """Nombre apto para archivo/tag de imagen: minúsculas, no-alfanumérico -> '-'."""
    return re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-")


def por_nombre(nombre: str) -> ModeloEvaluado2026:
    for modelo in MODELOS_2026:
        if modelo.nombre == nombre:
            return modelo
    raise ValueError(
        f"No existe el modelo {nombre!r} en el roster 2026. "
        f"Disponibles: {', '.join(m.nombre for m in MODELOS_2026)}"
    )


def roster_activo() -> list[ModeloEvaluado2026]:
    """Los 12 modelos con activo=True, en orden de registro."""
    return [m for m in MODELOS_2026 if m.activo]


def roster_baseline_original() -> list[ModeloEvaluado2026]:
    """Los 4 modelos del paper original (Tabla 2 publicada), en orden de registro
    (Delta 05). Eje ORTOGONAL a `roster_activo()`: 3 de los 4 ya están adentro
    (son también las anclas de continuidad de RF19); el cuarto,
    `Qwen2.5-0.5B-Instruct`, es `baseline_original=True` con `activo=False` --
    completa la medición bajo el harness/prompt 2026 sin entrar al roster de 12."""
    return [m for m in MODELOS_2026 if m.baseline_original]


def por_tier(tier: Tier) -> list[ModeloEvaluado2026]:
    """Filtra el ROSTER ACTIVO (no el registro completo): F3 lo re-congela así."""
    return [m for m in roster_activo() if m.tier == tier]


def gated() -> list[ModeloEvaluado2026]:
    """Los modelos que requieren $HF_TOKEN para descargarse, en orden de registro."""
    return [m for m in MODELOS_2026 if m.gated]
