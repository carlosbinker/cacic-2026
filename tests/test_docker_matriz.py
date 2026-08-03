"""Tests de la orquestación Docker y del contrato de documentación de pines."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))

from build_all import comando_build, tag_imagen  # noqa: E402
from models_2026 import BASELINE_TRANSFORMERS, MODELOS_2026, roster_activo, slug  # noqa: E402
from run_sweep import comando_run  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def test_tag_por_modelo_es_unico_y_usa_el_slug():
    tags = [tag_imagen(m) for m in MODELOS_2026]
    assert len(set(tags)) == 14
    assert tag_imagen(MODELOS_2026[0]) == f"slm-domotica-2026:{slug(MODELOS_2026[0].nombre)}"


def test_comando_build_pasa_el_pin_como_build_arg():
    modelo = MODELOS_2026[0]
    cmd = comando_build(modelo, RAIZ)
    assert "--build-arg" in cmd
    i = cmd.index("--build-arg")
    assert cmd[i + 1] == f"TRANSFORMERS_PIN={modelo.transformers_pin}"
    assert "-f" in cmd and cmd[cmd.index("-f") + 1].endswith("Dockerfile.modelo")
    assert tag_imagen(modelo) in cmd


def test_comando_run_respeta_los_limites_de_hardware_del_paper():
    modelo = MODELOS_2026[0]
    cmd = comando_run(modelo, RAIZ)
    assert "--memory=8g" in cmd and "--cpus=2" in cmd
    assert "--rm" in cmd


def test_comando_run_monta_datos_y_cache_compartida():
    cmd = comando_run(MODELOS_2026[0], RAIZ)
    montajes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert any(m.endswith(":/app/data") for m in montajes)
    assert any(m.endswith(":/app/.hf_cache") for m in montajes)


def test_comando_run_invoca_el_harness_con_el_nombre_del_modelo():
    modelo = MODELOS_2026[3]
    cmd = comando_run(modelo, RAIZ)
    assert cmd[-3:] == ["src/run_sweep_2026.py", "--modelo", modelo.nombre]


def test_no_hay_credenciales_en_capas_ni_login_interactivo():
    """F11: prohibido declarar HF_TOKEN en el Dockerfile, copiar .env, pasar el
    token por --build-arg o hacer login interactivo. NO prohibido: --env-file
    en el docker run de los dos modelos gated (se testea aparte)."""
    for ruta in (RAIZ / "docker").rglob("*"):
        if not ruta.is_file():
            continue
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"ARG\s+HF_TOKEN", texto), f"{ruta.name}: ARG HF_TOKEN"
        assert not re.search(r"ENV\s+HF_TOKEN", texto), f"{ruta.name}: ENV HF_TOKEN"
        assert not re.search(r"COPY\s+\.env", texto), f"{ruta.name}: COPY .env"
        bajo = texto.lower()
        assert "huggingface-cli login" not in bajo, f"{ruta.name}: login interactivo"
        assert "huggingface_hub.login(" not in bajo, f"{ruta.name}: huggingface_hub.login("


def test_docker_no_contiene_el_valor_del_token():
    """El patrón es estricto a propósito: hf_ + ~34 alfanuméricos es un token
    real; hf_repo_id, .hf_cache, HF_TOKEN y HF_HOME son identificadores
    legítimos del diseño y no deben matchear."""
    for ruta in (RAIZ / "docker").rglob("*"):
        if ruta.is_file():
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
            assert not re.search(r"hf_[A-Za-z0-9]{20,}", texto), f"{ruta.name} versiona un token"


def test_run_sweep_agrega_env_file_solo_a_los_gated():
    """F11: --env-file .env solo para los dos modelos gated; los otros 12 no."""
    gated = [m for m in MODELOS_2026 if m.gated]
    no_gated = [m for m in MODELOS_2026 if not m.gated]
    assert len(gated) == 2
    assert len(no_gated) == 12
    for m in gated:
        cmd = comando_run(m, RAIZ)
        assert "--env-file" in cmd
        i = cmd.index("--env-file")
        assert cmd[i + 1] == str(RAIZ / ".env")
    for m in no_gated:
        cmd = comando_run(m, RAIZ)
        assert "--env-file" not in cmd


def test_el_readme_documenta_la_matriz_completa():
    """RF5: cada modelo aparece en la tabla de docker/README.md."""
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for m in MODELOS_2026:
        assert m.nombre in readme, f"falta {m.nombre} en docker/README.md"


def test_todo_pin_divergente_esta_justificado_en_el_readme():
    """RF5: si un pin diverge del baseline, su motivo debe estar escrito."""
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for m in MODELOS_2026:
        if m.transformers_pin != BASELINE_TRANSFORMERS:
            assert m.motivo_pin, f"{m.nombre} diverge sin motivo_pin"
            assert m.transformers_pin in readme, f"falta el pin de {m.nombre}"
            assert m.motivo_pin[:30] in readme, f"falta el motivo de {m.nombre}"
        if m.trust_remote_code:
            assert re.search(rf"{re.escape(m.nombre)}.*trust_remote_code", readme, re.S | re.I), \
                f"{m.nombre} usa trust_remote_code sin documentarlo"


def test_build_all_selecciona_el_roster_activo_por_defecto():
    from build_all import seleccionar_modelos as seleccionar_build
    seleccionados = seleccionar_build(None)
    assert seleccionados == roster_activo()
    assert len(seleccionados) == 12


def test_run_sweep_selecciona_el_roster_activo_por_defecto():
    from run_sweep import seleccionar_modelos as seleccionar_run
    seleccionados = seleccionar_run(None)
    assert seleccionados == roster_activo()
    assert len(seleccionados) == 12


def test_build_all_pedir_un_modelo_excluido_falla_con_el_motivo():
    from build_all import seleccionar_modelos as seleccionar_build
    with pytest.raises(ValueError) as exc:
        seleccionar_build("gemma-3-270m-it")
    assert "excluido" in str(exc.value)
    assert "403" in str(exc.value) or "resolve" in str(exc.value)


def test_run_sweep_desde_un_modelo_excluido_falla_con_el_motivo():
    from run_sweep import seleccionar_modelos as seleccionar_run
    with pytest.raises(ValueError) as exc:
        seleccionar_run("Llama-3.2-1B-Instruct")
    assert "excluido" in str(exc.value)


def test_build_all_pedir_un_modelo_activo_sigue_funcionando():
    from build_all import seleccionar_modelos as seleccionar_build
    seleccionados = seleccionar_build("SmolLM2-360M-Instruct")
    assert [m.nombre for m in seleccionados] == ["SmolLM2-360M-Instruct"]


def test_el_readme_documenta_la_matriz_del_roster_activo_con_necesidad():
    """RF5 + F11: los 12 del roster activo estan en la matriz con necesario/heredado."""
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for m in roster_activo():
        assert m.nombre in readme, f"falta {m.nombre} en la matriz de docker/README.md"
    necesarios = {m.nombre for m in roster_activo() if m.motivo_pin}
    assert necesarios == {
        "LFM2.5-230M", "LFM2.5-350M", "Qwen3.5-0.8B", "Qwen3.5-2B", "granite-4.0-350m",
    }


def test_el_readme_documenta_las_dos_exclusiones():
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for m in MODELOS_2026:
        if not m.activo:
            assert m.nombre in readme, f"falta la exclusion de {m.nombre}"
            assert "403" in readme
