"""Tests del registro de modelos 2026 (F3 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import (  # noqa: E402
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    gated,
    por_nombre,
    por_tier,
    slug,
)

DESCARTADOS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
}


def test_el_roster_tiene_catorce_modelos():
    assert len(MODELOS_2026) == 14


def test_no_hay_modelos_qwen25():
    repos = {m.hf_repo_id for m in MODELOS_2026}
    assert repos & DESCARTADOS == set()


def test_exactamente_dos_modelos_gated():
    gateados = {m.hf_repo_id for m in MODELOS_2026 if m.gated}
    assert gateados == {"google/gemma-3-270m-it", "meta-llama/Llama-3.2-1B-Instruct"}
    assert {m.hf_repo_id for m in gated()} == gateados


def test_nombres_y_repos_son_unicos():
    assert len({m.nombre for m in MODELOS_2026}) == 14
    assert len({m.hf_repo_id for m in MODELOS_2026}) == 14


def test_siete_por_tier():
    assert len(por_tier("sub-1B")) == 7
    assert len(por_tier("1-2B")) == 7


def test_smollm2_aporta_los_dos_tamanos_del_paper_original():
    repos = {m.hf_repo_id for m in MODELOS_2026}
    assert "HuggingFaceTB/SmolLM2-360M-Instruct" in repos
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in repos


def test_los_slugs_son_unicos_y_aptos_para_nombre_de_archivo():
    slugs = [slug(m.nombre) for m in MODELOS_2026]
    assert len(set(slugs)) == 14
    for s in slugs:
        assert s and all(c.isalnum() or c == "-" for c in s)
        assert not s.startswith("-") and not s.endswith("-")


def test_slug_normaliza_puntos_y_mayusculas():
    assert slug("Qwen3.5-0.8B") == "qwen3-5-0-8b"
    assert slug("granite-4.0-h-1b") == "granite-4-0-h-1b"


def test_invariante_motivo_pin():
    """motivo_pin no vacío si y solo si el pin diverge del baseline (F3)."""
    for m in MODELOS_2026:
        diverge = m.transformers_pin != BASELINE_TRANSFORMERS
        assert bool(m.motivo_pin) == diverge, (
            f"{m.nombre}: pin={m.transformers_pin!r} motivo={m.motivo_pin!r}"
        )


def test_estado_inicial_sin_divergencias():
    """Arranca sin pines divergentes; solo el subtask 04 puede cambiarlo."""
    assert all(m.transformers_pin == BASELINE_TRANSFORMERS for m in MODELOS_2026)
    assert all(m.trust_remote_code is False for m in MODELOS_2026)


def test_por_nombre_encuentra_y_falla_bien():
    assert por_nombre("Qwen3.5-2B").hf_repo_id == "Qwen/Qwen3.5-2B"
    with pytest.raises(ValueError, match="inexistente"):
        por_nombre("inexistente")
