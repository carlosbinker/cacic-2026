"""Tests del registro de modelos 2026 (F3 del índice)."""

import json
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
    "Qwen/Qwen2.5-0.5B-Instruct",   # superado por Qwen3.5-0.8B
}
# Qwen/Qwen2.5-1.5B-Instruct YA NO esta descartado: el Delta 2026-08-04 lo
# reincorporo al roster activo (indice §2.1, RF19).
INEXISTENTES = {
    "Qwen/Qwen2.5-1.7B-Instruct",   # HTTP 404: el usuario lo pidio asi, no existe
}


def test_el_registro_tiene_quince_modelos():
    assert len(MODELOS_2026) == 15


def test_no_hay_modelos_descartados_ni_inexistentes():
    repos = {m.hf_repo_id for m in MODELOS_2026}
    assert repos & DESCARTADOS == set()
    assert repos & INEXISTENTES == set(), (
        "Qwen/Qwen2.5-1.7B-Instruct no existe (HTTP 404); el ID correcto es "
        "Qwen/Qwen2.5-1.5B-Instruct"
    )


def test_exactamente_dos_modelos_gated():
    gateados = {m.hf_repo_id for m in MODELOS_2026 if m.gated}
    assert gateados == {"google/gemma-3-270m-it", "meta-llama/Llama-3.2-1B-Instruct"}
    assert {m.hf_repo_id for m in gated()} == gateados


def test_nombres_y_repos_son_unicos():
    assert len({m.nombre for m in MODELOS_2026}) == 15
    assert len({m.hf_repo_id for m in MODELOS_2026}) == 15


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
    assert len(set(slugs)) == 15
    for s in slugs:
        assert s and all(c.isalnum() or c == "-" for c in s)
        assert not s.startswith("-") and not s.endswith("-")


def test_slug_del_qwen25_reincorporado():
    assert slug("Qwen2.5-1.5B-Instruct") == "qwen2-5-1-5b-instruct"


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


def test_dos_grupos_de_version_nueve_y_tres_en_el_roster_activo():
    """F3 (b) amendado el 2026-08-04: el intercambio movio una fila de B a A."""
    activos = roster_activo()
    grupo_a = [m for m in activos if m.transformers_pin == BASELINE_TRANSFORMERS]
    grupo_b = [m for m in activos if m.transformers_pin == TRANSFORMERS_5X]
    assert len(grupo_a) == 9
    assert len(grupo_b) == 3
    assert len(grupo_a) + len(grupo_b) == len(activos) == 12


def test_dos_grupos_de_version_once_y_cuatro_en_el_registro():
    """F3 (b): sobre el REGISTRO son 11 en A (9 activos + 2 gated) y 4 en B."""
    grupo_a = [m for m in MODELOS_2026 if m.transformers_pin == BASELINE_TRANSFORMERS]
    grupo_b = [m for m in MODELOS_2026 if m.transformers_pin == TRANSFORMERS_5X]
    assert len(grupo_a) == 11
    assert len(grupo_b) == 4


def test_el_registro_completo_tiene_quince_y_dos_gated():
    """F3 (c): el conteo de gated es del REGISTRO y NO cambio con el Delta 03."""
    assert len(MODELOS_2026) == 15
    gateados = {m.hf_repo_id for m in MODELOS_2026 if m.gated}
    assert gateados == {"google/gemma-3-270m-it", "meta-llama/Llama-3.2-1B-Instruct"}
    assert {m.hf_repo_id for m in gated()} == gateados


def test_trece_no_gated_en_el_registro():
    """12 activos + Qwen3.5-2B (excluido, no gated)."""
    assert len([m for m in MODELOS_2026 if not m.gated]) == 13


def test_roster_activo_sigue_en_doce_seis_por_tier_y_cero_gated():
    """F3 (d): el intercambio es a conteo constante (salio y entro un 1-2B)."""
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


def test_qwen25_15b_esta_activo_en_el_grupo_a_con_pin_heredado():
    """Delta 2026-08-04: entra al roster activo, grupo A por sonda propia
    (PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template), pin HEREDADO."""
    m = por_nombre("Qwen2.5-1.5B-Instruct")
    assert m.hf_repo_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert m.params_b == 1.54           # igual que la Tabla 1 publicada
    assert m.tier == "1-2B"
    assert m.transformers_pin == BASELINE_TRANSFORMERS
    assert m.gated is False
    assert m.trust_remote_code is False
    assert m.activo is True
    assert m.motivo_exclusion == ""
    assert m.motivo_pin == "", "su pin es heredado, no necesario: la sonda 5.14.1 no se corrio"


def test_qwen35_2b_esta_excluido_con_motivo_y_conserva_su_pin_necesario():
    """F3 (g): motivo_exclusion y motivo_pin son ortogonales."""
    m = por_nombre("Qwen3.5-2B")
    assert m.activo is False
    assert m.gated is False, "esta excluido pero NO es gated"
    assert m.transformers_pin == TRANSFORMERS_5X
    assert m.motivo_pin, "su pin del grupo B sigue siendo necesario (falla bajo 4.57.6)"
    assert m.motivo_exclusion
    bajo = m.motivo_exclusion.lower()
    assert "no verificada" in bajo or "nunca se obtuvo" in bajo
    # Redaccion taxativa: no se puede afirmar que falla bajo 5.14.1.
    assert "falla bajo 5.14.1" not in bajo
    assert "falla bajo transformers 5" not in bajo


def test_los_tres_excluidos_y_sus_dos_causas():
    excluidos = {m.nombre for m in MODELOS_2026 if not m.activo}
    assert excluidos == {"gemma-3-270m-it", "Llama-3.2-1B-Instruct", "Qwen3.5-2B"}
    for m in MODELOS_2026:
        if not m.activo and m.gated:
            assert "403" in m.motivo_exclusion or "resolve" in m.motivo_exclusion
        if not m.activo and not m.gated:
            assert "5.14.1" in m.motivo_exclusion


def test_pines_necesarios_cinco_en_el_registro_cuatro_en_el_roster():
    """F3 (g): todo test que afirme el conjunto de necesarios dice sobre cual cuenta."""
    necesarios_registro = {m.nombre for m in MODELOS_2026 if m.motivo_pin}
    assert necesarios_registro == {
        "LFM2.5-230M", "LFM2.5-350M", "Qwen3.5-0.8B", "Qwen3.5-2B", "granite-4.0-350m",
    }
    necesarios_activos = {m.nombre for m in roster_activo() if m.motivo_pin}
    assert necesarios_activos == {
        "LFM2.5-230M", "LFM2.5-350M", "Qwen3.5-0.8B", "granite-4.0-350m",
    }


def test_los_tres_anclas_de_continuidad_con_el_paper_publicado():
    """RF19: los modelos presentes en los dos estudios pasan de 2 a 3, y sus
    nombres coinciden textualmente con las claves del resumen publicado (F0)."""
    publicado = json.loads(
        (Path(__file__).resolve().parent.parent
         / "data" / "resultados_experimento_resumen.json").read_text(encoding="utf-8")
    )
    nombres_publicados = {f["modelo"] for f in publicado}
    anclas = {m.nombre for m in roster_activo()} & nombres_publicados
    assert anclas == {
        "SmolLM2-360M-Instruct", "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct",
    }
    # El params_b del registro coincide con el publicado, para que la comparacion
    # sea del mismo modelo y no de una variante de otro tamaño.
    por_nombre_publicado = {f["modelo"]: f for f in publicado}
    for nombre in anclas:
        assert por_nombre(nombre).params_b == por_nombre_publicado[nombre]["params_b"]


def test_roster_activo_es_subconjunto_ordenado_del_registro():
    activos = roster_activo()
    indices = [MODELOS_2026.index(m) for m in activos]
    assert indices == sorted(indices), "roster_activo() no respeta el orden del registro"


def test_por_nombre_encuentra_y_falla_bien():
    assert por_nombre("Qwen3.5-2B").hf_repo_id == "Qwen/Qwen3.5-2B"
    with pytest.raises(ValueError, match="inexistente"):
        por_nombre("inexistente")
