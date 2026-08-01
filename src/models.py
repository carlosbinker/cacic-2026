"""
Los cuatro modelos evaluados (Tabla 1 del paper), con sus identificadores
reales de HuggingFace Hub. Los cuatro son open-weight, sin gating ni
licencia restrictiva, tal como especifica la Sección 3.1.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeloEvaluado:
    nombre: str          # nombre tal como aparece en el paper
    hf_repo_id: str       # identificador en HuggingFace Hub
    params_b: float       # parámetros, en miles de millones


MODELOS = [
    ModeloEvaluado(
        nombre="SmolLM2-360M-Instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        params_b=0.36,
    ),
    ModeloEvaluado(
        nombre="Qwen2.5-0.5B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.494,
    ),
    ModeloEvaluado(
        nombre="Qwen2.5-1.5B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.54,
    ),
    ModeloEvaluado(
        nombre="SmolLM2-1.7B-Instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        params_b=1.71,
    ),
]
