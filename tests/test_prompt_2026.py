"""Tests del prompt 2026 y del despacho de modo de prompting (F2 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt import SYSTEM_PROMPT_PAPER  # noqa: E402
from prompt_2026 import (  # noqa: E402
    CLAUSULA_TAXATIVA,
    SYSTEM_PROMPT_2026,
    construir_entrada,
)


def test_el_prompt_2026_extiende_al_del_paper_sin_alterarlo():
    assert SYSTEM_PROMPT_2026.startswith(SYSTEM_PROMPT_PAPER)
    assert len(SYSTEM_PROMPT_2026) > len(SYSTEM_PROMPT_PAPER)


def test_la_clausula_taxativa_esta_incluida():
    assert CLAUSULA_TAXATIVA in SYSTEM_PROMPT_2026


def test_la_clausula_declara_los_campos_categoricos_cerrados():
    for campo in ('"intent"', '"dispositivo"', '"ubicacion"', '"unidad"'):
        assert campo in CLAUSULA_TAXATIVA
    assert "CATEGORICOS CERRADOS" in CLAUSULA_TAXATIVA


def test_la_clausula_declara_los_sinonimos_violacion_de_formato():
    texto = CLAUSULA_TAXATIVA.lower()
    assert "unicos outputs aceptados" in texto
    assert "sinonimo" in texto
    assert "violacion del formato" in texto


def test_la_clausula_da_un_ejemplo_concreto_de_sinonimo_prohibido():
    assert "televisor" in CLAUSULA_TAXATIVA and "tv" in CLAUSULA_TAXATIVA


def test_el_segundo_ejemplo_fewshot_cubre_ajuste_sin_valor():
    assert "Ejemplo 2" in SYSTEM_PROMPT_2026
    assert '"valor": null, "unidad": null' in SYSTEM_PROMPT_2026


class TokenizerConChat:
    """Doble mínimo de un tokenizer instruct: expone chat_template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def __init__(self):
        self.mensajes_recibidos = None
        self.kwargs_recibidos = None

    def apply_chat_template(self, mensajes, **kwargs):
        self.mensajes_recibidos = mensajes
        self.kwargs_recibidos = kwargs
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


def test_construir_entrada_arma_system_y_user():
    tok = TokenizerConChat()
    entrada = construir_entrada(tok, "Prendé la luz del living.")
    assert "input_ids" in entrada
    assert tok.mensajes_recibidos[0]["role"] == "system"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_2026
    assert tok.mensajes_recibidos[1]["role"] == "user"
    assert tok.mensajes_recibidos[1]["content"] == 'Comando: "Prendé la luz del living."'
    # return_dict=True es obligatorio: sin él generate() falla con AttributeError
    assert tok.kwargs_recibidos["return_dict"] is True
    assert tok.kwargs_recibidos["add_generation_prompt"] is True
    assert tok.kwargs_recibidos["return_tensors"] == "pt"


def test_construir_entrada_rechaza_comando_vacio():
    with pytest.raises(ValueError, match="comando vacío"):
        construir_entrada(TokenizerConChat(), "   ")


# --------------------------------------------------------------------------
# Falla explícita sin chat_template (revierte el fallback a raw completion,
# 2026-08-05): un tokenizer sin chat_template debe fallar con un ValueError
# claro, no con un AttributeError/TypeError opaco más abajo en la pila.
# --------------------------------------------------------------------------

def test_construir_entrada_falla_explicitamente_si_chat_template_es_none():
    class TokenizerBase:
        chat_template = None

        def apply_chat_template(self, mensajes, **kwargs):
            raise AssertionError("no debería llegar a llamarse")

    with pytest.raises(ValueError, match="chat_template"):
        construir_entrada(TokenizerBase(), "Prendé la luz.")


def test_construir_entrada_falla_explicitamente_si_chat_template_es_cadena_vacia():
    class TokenizerBase:
        chat_template = ""

        def apply_chat_template(self, mensajes, **kwargs):
            raise AssertionError("no debería llegar a llamarse")

    with pytest.raises(ValueError, match="chat_template"):
        construir_entrada(TokenizerBase(), "Prendé la luz.")


def test_construir_entrada_falla_explicitamente_si_falta_el_atributo():
    class SinAtributo:
        def apply_chat_template(self, mensajes, **kwargs):
            raise AssertionError("no debería llegar a llamarse")

    with pytest.raises(ValueError, match="chat_template"):
        construir_entrada(SinAtributo(), "Prendé la luz.")
