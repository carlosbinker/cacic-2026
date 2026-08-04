"""Tests de la corrida de CONTROL (Delta 04): mismo harness 2026, mismo
dataset, mismo orden, misma decodificación y mismo scoring que el barrido
principal -- la ÚNICA variable que cambia es el prompt de sistema, que pasa
de `SYSTEM_PROMPT_2026` a `SYSTEM_PROMPT_PAPER` (el original del paper,
importado tal cual de `src/prompt.py`, F0 congelado).

Ninguno de estos tests ejerce inferencia real (no se descarga ni carga
ningún modelo) ni escribe bajo `data/2026/`: todo el I/O de archivos pasa
por `tmp_path`, siguiendo la misma convención de
`tests/test_run_sweep_2026.py` y `tests/test_docker_matriz.py`.

Este archivo crece en tres etapas (Delta 04, tasks 1-3): primero la
infraestructura factorizada y el entrypoint de control (`src/
run_control_prompt_original.py`), luego el runner host-side
(`docker/run_control.py`) y por último el generador de comparación de tres
bandas (`src/comparar_control.py`).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))

from models_2026 import por_nombre, slug  # noqa: E402
from prompt import SYSTEM_PROMPT_PAPER  # noqa: E402
from prompt_2026 import SYSTEM_PROMPT_2026, construir_entrada_con_prompt  # noqa: E402

import run_sweep_2026  # noqa: E402
import run_control_prompt_original as rcpo  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Dobles de tokenizer (mismos que tests/test_prompt_2026.py)
# --------------------------------------------------------------------------

class TokenizerConChat:
    """Doble mínimo de un tokenizer instruct: expone chat_template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def __init__(self):
        self.mensajes_recibidos = None
        self.kwargs_recibidos = None

    def apply_chat_template(self, mensajes, **kwargs):
        self.mensajes_recibidos = mensajes
        self.kwargs_recibidos = kwargs
        return {"input_ids": [[1, 2, 3]]}

    def __call__(self, texto, **kwargs):
        raise AssertionError("no se debe tokenizar en crudo un modelo con chat_template")


class TokenizerBase:
    """Doble mínimo de un modelo base: chat_template es None."""

    chat_template = None

    def __init__(self):
        self.texto_recibido = None

    def apply_chat_template(self, mensajes, **kwargs):
        raise AssertionError("no se debe usar chat_template en un modelo base")

    def __call__(self, texto, **kwargs):
        self.texto_recibido = texto
        return {"input_ids": [[4, 5]]}


# --------------------------------------------------------------------------
# Infraestructura factorizada (habilita la corrida de control sin copiarla)
# --------------------------------------------------------------------------

def test_construir_entrada_2026_sigue_usando_el_prompt_2026_por_defecto():
    """La factorización no puede cambiar el comportamiento del barrido principal."""
    from prompt_2026 import construir_entrada
    tok = TokenizerConChat()
    construir_entrada(tok, "Prendé la luz.")
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_2026


def test_construir_entrada_con_prompt_permite_sustituir_el_sistema():
    tok = TokenizerConChat()
    entrada, modo = construir_entrada_con_prompt(tok, "Prendé la luz.", SYSTEM_PROMPT_PAPER)
    assert modo == "chat_template"
    assert tok.mensajes_recibidos[0]["role"] == "system"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_PAPER
    assert tok.mensajes_recibidos[0]["content"] != SYSTEM_PROMPT_2026
    assert tok.mensajes_recibidos[1]["content"] == 'Comando: "Prendé la luz."'


def test_construir_entrada_con_prompt_modo_raw_usa_el_prompt_dado():
    tok = TokenizerBase()
    entrada, modo = construir_entrada_con_prompt(tok, "Apagá la tele.", SYSTEM_PROMPT_PAPER)
    assert modo == "raw_completion"
    assert tok.texto_recibido.startswith(SYSTEM_PROMPT_PAPER)


def test_construir_entrada_con_prompt_rechaza_comando_vacio():
    with pytest.raises(ValueError, match="comando vacío"):
        construir_entrada_con_prompt(TokenizerConChat(), "   ", SYSTEM_PROMPT_PAPER)


def test_evaluar_modelo_acepta_un_constructor_de_entrada_alternativo(monkeypatch):
    """`run_sweep_2026.evaluar_modelo` debe poder recibir un builder de entrada
    distinto sin tocar nada más del pipeline (mismo dataset, mismo scoring)."""
    llamadas = []

    class FakeCtx:
        modelo = por_nombre("SmolLM2-360M-Instruct")
        tokenizer = object()
        transformers_version = "4.57.6"

    def fake_abrir_contexto(_modelo):
        return FakeCtx()

    def fake_generar_respuesta(_ctx, _entrada):
        return (
            '{"intent": "encender", "dispositivo": "luz", "ubicacion": "living", '
            '"valor": null, "unidad": null}',
            0.01,
        )

    def constructor_de_prueba(_tokenizer, comando):
        llamadas.append(comando)
        return {"input_ids": [[1]]}, "chat_template"

    monkeypatch.setattr(run_sweep_2026, "_abrir_contexto", fake_abrir_contexto)
    monkeypatch.setattr(run_sweep_2026, "_generar_respuesta", fake_generar_respuesta)

    dataset = run_sweep_2026.cargar_dataset()
    filas = run_sweep_2026.evaluar_modelo(
        por_nombre("SmolLM2-360M-Instruct"), dataset, construir_entrada=constructor_de_prueba
    )
    assert len(llamadas) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert len(filas) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert all(f["modo_prompting"] == "chat_template" for f in filas)


# --------------------------------------------------------------------------
# Task 1: src/run_control_prompt_original.py
# --------------------------------------------------------------------------

def test_anclas_control_son_exactamente_las_tres_de_continuidad():
    assert set(rcpo.ANCLAS_CONTROL) == {
        "SmolLM2-360M-Instruct", "SmolLM2-1.7B-Instruct", "Qwen2.5-1.5B-Instruct",
    }
    assert len(rcpo.ANCLAS_CONTROL) == 3


def test_ruta_control_usa_el_slug_bajo_control_prompt_original():
    ruta = rcpo.ruta_control("Qwen2.5-1.5B-Instruct")
    assert ruta.name == f"{slug('Qwen2.5-1.5B-Instruct')}.csv"
    assert ruta.parent.name == "control_prompt_original"
    assert ruta.parent.parent.name == "2026"


def test_verificar_es_ancla_rechaza_un_modelo_fuera_de_los_tres():
    otro = por_nombre("LFM2.5-230M")
    with pytest.raises(ValueError, match="no es uno de los tres anclajes"):
        rcpo.verificar_es_ancla(otro)


def test_verificar_es_ancla_acepta_los_tres_anclajes():
    for nombre in rcpo.ANCLAS_CONTROL:
        rcpo.verificar_es_ancla(por_nombre(nombre))  # no debe lanzar


def test_construir_entrada_original_usa_el_prompt_del_paper_no_el_2026():
    tok = TokenizerConChat()
    entrada, modo = rcpo.construir_entrada_original(tok, "Prendé la luz.")
    assert modo == "chat_template"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_PAPER
    assert tok.mensajes_recibidos[0]["content"] != SYSTEM_PROMPT_2026


def test_main_escribe_el_csv_de_control_con_las_columnas_congeladas(monkeypatch, tmp_path):
    """CLI de punta a punta sin inferencia real: `evaluar_modelo` se reemplaza
    por un fake que arma filas con `armar_fila` real (mismo scoring que el
    barrido principal), y la escritura va a `tmp_path`, no a `data/2026/`."""
    ruta_tmp = tmp_path / "smollm2-360m-instruct.csv"
    monkeypatch.setattr(rcpo, "ruta_control", lambda nombre: ruta_tmp)

    dataset = run_sweep_2026.cargar_dataset()

    def fake_evaluar_modelo(modelo_, dataset_, construir_entrada=None):
        assert construir_entrada is rcpo.construir_entrada_original
        filas = []
        for idx, fila_ds in dataset_.iterrows():
            filas.append(run_sweep_2026.armar_fila(
                modelo=modelo_, idx=int(idx), comando=fila_ds["comando"],
                gt={"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
                    "valor": None, "unidad": None},
                texto_generado='{"intent": "encender", "dispositivo": "luz", '
                               '"ubicacion": "living", "valor": null, "unidad": null}',
                latencia_s=0.01, modo="chat_template", transformers_version="4.57.6",
            ))
        return filas

    monkeypatch.setattr(rcpo, "evaluar_modelo", fake_evaluar_modelo)
    monkeypatch.setattr(rcpo, "cargar_dataset", lambda: dataset)
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "SmolLM2-360M-Instruct",
    ])

    codigo = rcpo.main()
    assert codigo == 0
    df = pd.read_csv(ruta_tmp)
    assert list(df.columns) == run_sweep_2026.COLUMNAS_DETALLE
    assert len(df) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert sorted(df["idx"]) == list(range(run_sweep_2026.N_COMANDOS_ESPERADO))


def test_main_rechaza_un_modelo_fuera_de_los_tres_anclajes(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "LFM2.5-230M",
    ])
    with pytest.raises(ValueError, match="no es uno de los tres anclajes"):
        rcpo.main()


def test_main_saltea_si_el_csv_de_control_ya_esta_completo(monkeypatch, tmp_path):
    ruta_tmp = tmp_path / "ya-completo.csv"
    pd.DataFrame(
        {c: [None] * run_sweep_2026.N_COMANDOS_ESPERADO for c in run_sweep_2026.COLUMNAS_DETALLE}
        | {"idx": list(range(run_sweep_2026.N_COMANDOS_ESPERADO))}
    ).to_csv(ruta_tmp, index=False)

    monkeypatch.setattr(rcpo, "ruta_control", lambda nombre: ruta_tmp)

    def fallar_si_se_llama(*_a, **_k):
        raise AssertionError("no debería evaluarse: el CSV de control ya está completo")

    monkeypatch.setattr(rcpo, "evaluar_modelo", fallar_si_se_llama)
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "SmolLM2-360M-Instruct",
    ])
    assert rcpo.main() == 0


def test_el_modulo_de_control_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "run_control_prompt_original.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea
