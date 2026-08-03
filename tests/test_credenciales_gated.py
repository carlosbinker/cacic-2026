"""Invariante del Delta 01 / F11: si la corrida incluye modelos gated y falta
el token, el barrido aborta ANTES de construir o descargar nada, con un
mensaje que jamás imprime el valor del token."""

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from models_2026 import MODELOS_2026, gated, por_nombre  # noqa: E402

# Valor ficticio deliberadamente sin la forma `hf_` + 20+ alfanuméricos, para
# que el escaneo de secretos (AC12) no lo tome por un token real.
TOKEN_FICTICIO = "valor-de-prueba-no-es-un-token"


def _cargar_run_sweep():
    """`docker/` no es un paquete importable: se carga el módulo por ruta."""
    ruta = RAIZ / "docker" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("run_sweep_docker", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def run_sweep():
    return _cargar_run_sweep()


def test_aborta_si_no_existe_el_archivo_env(run_sweep, tmp_path):
    with pytest.raises(ValueError) as exc:
        run_sweep.validar_credenciales(gated(), tmp_path)
    assert ".env" in str(exc.value)


def test_aborta_si_el_token_esta_vacio(run_sweep, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        run_sweep.validar_credenciales(gated(), tmp_path)
    assert "HF_TOKEN" in str(exc.value)


def test_aborta_si_el_token_no_esta_declarado(run_sweep, tmp_path):
    (tmp_path / ".env").write_text("OTRA_COSA=1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_sweep.validar_credenciales(gated(), tmp_path)


@pytest.mark.parametrize("nombre", ["gemma-3-270m-it", "Llama-3.2-1B-Instruct"])
def test_un_solo_modelo_gated_ya_dispara_el_aborto(run_sweep, tmp_path, nombre):
    with pytest.raises(ValueError):
        run_sweep.validar_credenciales([por_nombre(nombre)], tmp_path)


def test_los_doce_no_gated_corren_sin_credenciales(run_sweep, tmp_path):
    no_gated = [m for m in MODELOS_2026 if not m.gated]
    assert len(no_gated) == 12
    assert run_sweep.validar_credenciales(no_gated, tmp_path) is None


def test_no_aborta_con_el_token_presente(run_sweep, tmp_path):
    (tmp_path / ".env").write_text(f"HF_TOKEN={TOKEN_FICTICIO}\n", encoding="utf-8")
    assert run_sweep.validar_credenciales(list(MODELOS_2026), tmp_path) is None


def test_leer_el_token_no_lo_imprime(run_sweep, tmp_path, capsys):
    ruta_env = tmp_path / ".env"
    ruta_env.write_text(f"HF_TOKEN={TOKEN_FICTICIO}\n", encoding="utf-8")
    assert run_sweep._hf_token_de_env(ruta_env) == TOKEN_FICTICIO
    capturado = capsys.readouterr()
    assert TOKEN_FICTICIO not in capturado.out
    assert TOKEN_FICTICIO not in capturado.err


def test_el_mensaje_de_aborto_no_filtra_el_token(run_sweep, tmp_path):
    (tmp_path / ".env").write_text(f"HF_TOKEN=   \nOTRA={TOKEN_FICTICIO}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        run_sweep.validar_credenciales(gated(), tmp_path)
    assert TOKEN_FICTICIO not in str(exc.value)
