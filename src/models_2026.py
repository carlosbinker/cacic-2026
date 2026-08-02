"""Registro de los 14 modelos evaluados en el experimento 2026 (F3 del índice).

Puro dato + validación: sin dependencias de `torch`/`transformers`, para que se
pueda importar y testear en cualquier contexto, incluidos los contenedores
mínimos del subtask 04. Declara el flag `gated` únicamente; nunca lee ni
imprime el valor de ningún token de HuggingFace.
"""

import re
from dataclasses import dataclass
from typing import Literal

Tier = Literal["sub-1B", "1-2B"]

BASELINE_TRANSFORMERS: str = "transformers>=4.57.0"


@dataclass(frozen=True)
class ModeloEvaluado2026:
    """Un modelo del roster 2026, con su plomería de reproducibilidad."""

    nombre: str  # clave única; valor de la columna `modelo` en todos los CSV
    hf_repo_id: str
    params_b: float
    tier: Tier
    transformers_pin: str  # spec pip exacto usado en la imagen de ese modelo
    trust_remote_code: bool
    gated: bool  # requiere licencia aceptada + $HF_TOKEN para descargarse
    motivo_pin: str  # "" si transformers_pin == BASELINE_TRANSFORMERS; si no, el porqué


MODELOS_2026: list[ModeloEvaluado2026] = [
    ModeloEvaluado2026(
        nombre="LFM2.5-230M",
        hf_repo_id="LiquidAI/LFM2.5-230M",
        params_b=0.23,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
    ),
    ModeloEvaluado2026(
        nombre="LFM2.5-350M",
        hf_repo_id="LiquidAI/LFM2.5-350M",
        params_b=0.35,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
    ),
    ModeloEvaluado2026(
        nombre="granite-4.0-350m",
        hf_repo_id="ibm-granite/granite-4.0-350m",
        params_b=0.35,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
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
    ),
    ModeloEvaluado2026(
        nombre="Qwen3.5-0.8B",
        hf_repo_id="Qwen/Qwen3.5-0.8B",
        params_b=0.8,
        tier="sub-1B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
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
    ),
    ModeloEvaluado2026(
        nombre="Qwen3.5-2B",
        hf_repo_id="Qwen/Qwen3.5-2B",
        params_b=2.0,
        tier="1-2B",
        transformers_pin=BASELINE_TRANSFORMERS,
        trust_remote_code=False,
        gated=False,
        motivo_pin="",
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


def por_tier(tier: Tier) -> list[ModeloEvaluado2026]:
    return [m for m in MODELOS_2026 if m.tier == tier]


def gated() -> list[ModeloEvaluado2026]:
    """Los modelos que requieren $HF_TOKEN para descargarse, en orden de roster."""
    return [m for m in MODELOS_2026 if m.gated]
