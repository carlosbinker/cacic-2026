"""Tests de la orquestación Docker y del contrato de documentación de pines."""

import json
import re
import subprocess
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
    assert len(set(tags)) == 15
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


def test_comando_run_fija_los_nucleos_con_cpuset():
    """F11/RNF6: --cpus=2 es una cuota de CFS; --cpuset-cpus fija los nucleos."""
    from run_sweep import CPUSET_POR_DEFECTO
    cmd = comando_run(roster_activo()[0], RAIZ)
    assert f"--cpuset-cpus={CPUSET_POR_DEFECTO}" in cmd
    assert CPUSET_POR_DEFECTO == "0-1"


def test_el_cpuset_es_el_mismo_en_las_doce_invocaciones():
    """Lo que hace comparable la latencia es que sea el MISMO par en las 12."""
    valores = set()
    for m in roster_activo():
        cmd = comando_run(m, RAIZ)
        valores |= {a for a in cmd if a.startswith("--cpuset-cpus=")}
    assert len(valores) == 1, valores


def test_el_cpuset_se_puede_sobreescribir_por_host():
    cmd = comando_run(roster_activo()[0], RAIZ, cpuset="2-3")
    assert "--cpuset-cpus=2-3" in cmd
    assert "--cpuset-cpus=0-1" not in cmd


def test_ninguna_invocacion_pide_gpu():
    for m in roster_activo():
        cmd = " ".join(comando_run(m, RAIZ))
        for prohibido in ("--gpus", "--device", "nvidia"):
            assert prohibido not in cmd


def test_ejecutar_barrido_no_corta_al_primer_fallo(monkeypatch, tmp_path):
    """F12.1 / RF20c: un modelo que falla no puede bloquear a los 11 restantes.
    El usuario no esta disponible para desbloquear el barrido."""
    import run_sweep

    corridos = []

    class Resultado:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stderr = "boom textual" if returncode else ""

    def falso_run(cmd, **kwargs):
        nombre = cmd[cmd.index("--modelo") + 1]
        corridos.append(nombre)
        # El segundo modelo del roster falla; el resto anda.
        return Resultado(1 if nombre == roster_activo()[1].nombre else 0)

    monkeypatch.setattr(run_sweep.subprocess, "run", falso_run)
    monkeypatch.setattr(run_sweep, "FALLOS_PATH", tmp_path / "fallos_barrido.json")

    codigo = run_sweep.ejecutar_barrido(roster_activo(), force=False, dry_run=False)

    assert [m.nombre for m in roster_activo()] == corridos, "se salteo algun modelo"
    assert codigo == 1, "tiene que salir 1 si hubo al menos un fallo"

    fallos = json.loads((tmp_path / "fallos_barrido.json").read_text(encoding="utf-8"))
    assert len(fallos) == 1
    assert fallos[0]["modelo"] == roster_activo()[1].nombre
    assert set(fallos[0]) == {
        "modelo", "hf_repo_id", "transformers_pin",
        "codigo_salida", "error_textual", "momento_iso",
    }


def test_ejecutar_barrido_sin_fallos_escribe_lista_vacia(monkeypatch, tmp_path):
    """F5: el archivo SIEMPRE existe, para no confundir 'no hubo fallos' con
    'no se registro'."""
    import run_sweep

    class Ok:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(run_sweep.subprocess, "run", lambda cmd, **kw: Ok())
    monkeypatch.setattr(run_sweep, "FALLOS_PATH", tmp_path / "fallos_barrido.json")
    assert run_sweep.ejecutar_barrido(roster_activo(), force=False, dry_run=False) == 0
    assert json.loads((tmp_path / "fallos_barrido.json").read_text(encoding="utf-8")) == []


def test_dry_run_no_escribe_el_archivo_de_fallos(monkeypatch, tmp_path):
    import run_sweep
    monkeypatch.setattr(run_sweep, "FALLOS_PATH", tmp_path / "fallos_barrido.json")
    assert run_sweep.ejecutar_barrido(roster_activo(), force=False, dry_run=True) == 0
    assert not (tmp_path / "fallos_barrido.json").exists()


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


def test_ningun_dockerfile_trackeado_tiene_credenciales_ni_login():
    """F11 sobre TODO el árbol trackeado, no solo docker/: un Dockerfile puede
    aparecer en cualquier ubicación (hay uno legacy en la raíz). Cierra el
    hueco de cobertura que deja el Check C4 de TEST_PLAN.md, que está acotado
    a docker/, src/ y scripts/ para no autofallarse contra su propia
    documentación (*.md, tests/)."""
    salida = subprocess.run(
        ["git", "ls-files", "*Dockerfile*"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout
    rutas = [RAIZ / linea for linea in salida.splitlines() if linea]
    assert rutas, "no se encontró ningún Dockerfile trackeado"
    for ruta in rutas:
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"ARG\s+HF_TOKEN", texto), f"{ruta}: ARG HF_TOKEN"
        assert not re.search(r"ENV\s+HF_TOKEN", texto), f"{ruta}: ENV HF_TOKEN"
        assert not re.search(r"COPY\s+\.env", texto), f"{ruta}: COPY .env"
        assert not re.search(r"--build-arg[= ]*HF_TOKEN", texto), f"{ruta}: --build-arg HF_TOKEN"
        bajo = texto.lower()
        assert "huggingface-cli login" not in bajo, f"{ruta}: login interactivo"
        assert "huggingface_hub.login(" not in bajo, f"{ruta}: huggingface_hub.login("


def test_docker_no_contiene_el_valor_del_token():
    """El patrón es estricto a propósito: hf_ + ~34 alfanuméricos es un token
    real; hf_repo_id, .hf_cache, HF_TOKEN y HF_HOME son identificadores
    legítimos del diseño y no deben matchear."""
    for ruta in (RAIZ / "docker").rglob("*"):
        if ruta.is_file():
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
            assert not re.search(r"hf_[A-Za-z0-9]{20,}", texto), f"{ruta.name} versiona un token"


def test_run_sweep_agrega_env_file_solo_a_los_gated():
    """F11: --env-file solo para los 2 gated; los otros 13 no (12 activos + Qwen3.5-2B)."""
    gated_ = [m for m in MODELOS_2026 if m.gated]
    no_gated = [m for m in MODELOS_2026 if not m.gated]
    assert len(gated_) == 2
    assert len(no_gated) == 13
    for m in gated_:
        cmd = comando_run(m, RAIZ)
        assert "--env-file" in cmd
        assert cmd[cmd.index("--env-file") + 1] == str(RAIZ / ".env")
    for m in no_gated:
        assert "--env-file" not in comando_run(m, RAIZ)


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
    """RF5 + F11: los 12 del roster activo estan en la matriz. Los necesarios
    DENTRO del roster activo son 4 (F3 (g)): Qwen3.5-2B sigue siendo necesario
    pero ya no esta en el roster."""
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for m in roster_activo():
        assert m.nombre in readme, f"falta {m.nombre} en la matriz de docker/README.md"
    necesarios = {m.nombre for m in roster_activo() if m.motivo_pin}
    assert necesarios == {
        "LFM2.5-230M", "LFM2.5-350M", "Qwen3.5-0.8B", "granite-4.0-350m",
    }


def test_el_readme_documenta_las_tres_exclusiones_con_su_causa():
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    excluidos = [m for m in MODELOS_2026 if not m.activo]
    assert len(excluidos) == 3
    for m in excluidos:
        assert m.nombre in readme, f"falta la exclusion de {m.nombre}"
    assert "403" in readme                      # los dos gated
    assert "no verificada" in readme.lower()    # Qwen3.5-2B


def test_el_readme_registra_el_cpuset_usado():
    """AC17: el valor efectivo de --cpuset-cpus queda escrito, no solo en el codigo."""
    from run_sweep import CPUSET_POR_DEFECTO
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    assert f"--cpuset-cpus={CPUSET_POR_DEFECTO}" in readme


def test_el_readme_no_promete_reanudar_desde_un_modelo_excluido():
    """El ejemplo viejo `--desde "Qwen3.5-2B"` ahora falla con ValueError: si
    figura como uso normal, el README miente."""
    readme = (RAIZ / "docker" / "README.md").read_text(encoding="utf-8")
    for linea in readme.splitlines():
        if "--desde" in linea and "Qwen3.5-2B" in linea:
            assert "falla" in linea.lower() or "excluido" in linea.lower(), linea
