"""Tests del registro de modelos 2026 (F3 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import (  # noqa: E402
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    TRANSFORMERS_5X,
    gated,
    por_nombre,
    por_tier,
    roster_activo,
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


def test_seis_por_tier_en_el_roster_activo():
    """por_tier filtra el ROSTER ACTIVO (12), no el registro completo (14)."""
    assert len(por_tier("sub-1B")) == 6
    assert len(por_tier("1-2B")) == 6
    assert all(m.activo for m in por_tier("sub-1B") + por_tier("1-2B"))


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


def test_invariante_motivo_pin_es_necesidad_no_divergencia():
    """F3 (a): motivo_pin != "" syss el pin es NECESARIO. granite-4.0-350m esta
    en el baseline Y tiene motivo, porque sin ese pin acotado no corre."""
    necesarios = {
        "LFM2.5-230M", "LFM2.5-350M", "Qwen3.5-0.8B", "Qwen3.5-2B",
        "granite-4.0-350m",
    }
    for m in MODELOS_2026:
        tiene_motivo = bool(m.motivo_pin)
        assert tiene_motivo == (m.nombre in necesarios), (
            f"{m.nombre}: motivo_pin={m.motivo_pin!r}"
        )
    # Corolario: todo el que diverge del baseline tiene motivo (la reciproca no vale).
    for m in MODELOS_2026:
        if m.transformers_pin != BASELINE_TRANSFORMERS:
            assert m.motivo_pin, f"{m.nombre} diverge del baseline sin motivo_pin"


def test_los_pines_reflejan_los_dos_grupos_de_version():
    """Tras el Delta 02 ya no hay un unico pin de arranque: hay dos grupos,
    cada uno necesario para una parte del roster (F3, invariante b)."""
    pines = {m.transformers_pin for m in MODELOS_2026}
    assert pines == {BASELINE_TRANSFORMERS, TRANSFORMERS_5X}
    assert all(m.trust_remote_code is False for m in MODELOS_2026)


def test_dos_grupos_de_version_ocho_y_cuatro():
    """F3 (b): exactamente dos grupos, 8 en A y 4 en B sobre el roster activo."""
    activos = roster_activo()
    grupo_a = [m for m in activos if m.transformers_pin == BASELINE_TRANSFORMERS]
    grupo_b = [m for m in activos if m.transformers_pin == TRANSFORMERS_5X]
    assert len(grupo_a) == 8
    assert len(grupo_b) == 4
    assert len(grupo_a) + len(grupo_b) == len(activos) == 12


def test_el_registro_completo_tiene_catorce_y_dos_gated():
    """F3 (c): el conteo de gated es una propiedad del REGISTRO, no del roster activo."""
    assert len(MODELOS_2026) == 14
    gateados = {m.hf_repo_id for m in MODELOS_2026 if m.gated}
    assert gateados == {"google/gemma-3-270m-it", "meta-llama/Llama-3.2-1B-Instruct"}


def test_roster_activo_tiene_doce_seis_por_tier_y_cero_gated():
    """F3 (d)."""
    activos = roster_activo()
    assert len(activos) == 12
    assert sum(1 for m in activos if m.tier == "sub-1B") == 6
    assert sum(1 for m in activos if m.tier == "1-2B") == 6
    assert all(not m.gated for m in activos)


def test_activo_syss_motivo_exclusion_no_vacio():
    """F3 (e): activo == False syss motivo_exclusion != ""."""
    for m in MODELOS_2026:
        assert (not m.activo) == bool(m.motivo_exclusion), (
            f"{m.nombre}: activo={m.activo} motivo_exclusion={m.motivo_exclusion!r}"
        )


def test_los_dos_gated_estan_excluidos_del_roster_activo():
    excluidos = {m.hf_repo_id for m in MODELOS_2026 if not m.activo}
    assert excluidos == {"google/gemma-3-270m-it", "meta-llama/Llama-3.2-1B-Instruct"}
    for m in MODELOS_2026:
        if m.hf_repo_id in excluidos:
            assert "403" in m.motivo_exclusion or "resolve" in m.motivo_exclusion


def test_roster_activo_es_subconjunto_ordenado_del_registro():
    activos = roster_activo()
    indices = [MODELOS_2026.index(m) for m in activos]
    assert indices == sorted(indices), "roster_activo() no respeta el orden del registro"


def test_por_nombre_encuentra_y_falla_bien():
    assert por_nombre("Qwen3.5-2B").hf_repo_id == "Qwen/Qwen3.5-2B"
    with pytest.raises(ValueError, match="inexistente"):
        por_nombre("inexistente")
