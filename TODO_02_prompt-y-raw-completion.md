---
id: 02
title: Prompt taxativo 2026 y ruta de raw completion
depends_on: [01]
files:
  - src/prompt_2026.py
  - tests/test_prompt_2026.py
---

## Spec

Implementar el contrato **F2** del índice: el prompt de sistema 2026 (prompt del paper + cláusula taxativa sobre campos categóricos + segundo ejemplo few-shot) y la construcción de entrada que despacha entre `chat_template` y `raw_completion` **por capacidad del tokenizer**, nunca por ID de modelo.

Esto cubre **RF1** (cláusula taxativa: los valores entre paréntesis son los únicos outputs aceptados, los sinónimos son violación de formato) y **RF2** (los dos modelos base `LFM2.5-230M` y `LFM2.5-350M` no tienen `chat_template` y deben correr igual).

`SYSTEM_PROMPT_PAPER` se **importa** de `src/prompt.py`, no se copia: `src/prompt.py` está congelado por F0 y sigue siendo la fuente de verdad del texto publicado. `construir_prompt_usuario` también se reutiliza tal cual.

## Implementation plan

### Tarea 1 — Texto del prompt 2026 (TDD)

- [ ] Escribir el test que falla, `tests/test_prompt_2026.py` (primera mitad):

```python
"""Tests del prompt 2026 y del despacho de modo de prompting (F2 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt import SYSTEM_PROMPT_PAPER  # noqa: E402
from prompt_2026 import (  # noqa: E402
    CLAUSULA_TAXATIVA,
    RAW_TEMPLATE,
    SYSTEM_PROMPT_2026,
    construir_entrada,
    detectar_modo,
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


def test_raw_template_tiene_los_dos_huecos_y_termina_pidiendo_json():
    assert "{system}" in RAW_TEMPLATE and "{comando}" in RAW_TEMPLATE
    assert RAW_TEMPLATE.rstrip().endswith("JSON:")
```

- [ ] Correr y confirmar que **falla** por `ModuleNotFoundError: No module named 'prompt_2026'`:
  `pytest -q tests/test_prompt_2026.py`
- [ ] Implementar la primera mitad de `src/prompt_2026.py`, con los textos **exactos** congelados en F2:

```python
"""
Prompt de sistema 2026 y construcción de entrada por capacidad del tokenizer.

Extiende --- sin modificarlo --- el prompt publicado en la Sección 3.5 del
paper, que se importa de src/prompt.py (archivo congelado). Los dos agregados
son: (a) la cláusula taxativa sobre campos categóricos cerrados, que es un
cambio deliberado de protocolo respecto del estudio original, y (b) el segundo
ejemplo few-shot que el paper declara haber usado pero no transcribe.

Como el roster 2026 incluye dos modelos base (LFM2.5-230M y LFM2.5-350M) que
no exponen chat_template, la construcción de entrada despacha por capacidad y
no por identificador de modelo.
"""

from typing import Literal

from prompt import SYSTEM_PROMPT_PAPER, construir_prompt_usuario

ModoPrompting = Literal["chat_template", "raw_completion"]

CLAUSULA_TAXATIVA = (
    'IMPORTANTE: los campos "intent", "dispositivo", "ubicacion" y "unidad" '
    "son campos CATEGORICOS CERRADOS. Los valores listados entre parentesis "
    "mas arriba son, textualmente, los unicos outputs aceptados para esos "
    "campos: copialos exactamente como aparecen, sin tildes, sin mayusculas, "
    "sin plural y sin reformular. Usar un sinonimo, una traduccion o "
    "cualquier variante que no este literalmente en esa lista (por ejemplo "
    '"televisor" o "tele" en lugar de "tv") es una violacion del formato y '
    "la respuesta se considera incorrecta. Si ningun valor de la lista "
    "aplica, usa null."
)

_SEGUNDO_EJEMPLO = (
    'Ejemplo 2: Comando: "Bajale un poco a la luz del living."\n'
    '{"intent": "ajustar", "dispositivo": "luz", "ubicacion": "living", '
    '"valor": null, "unidad": null}'
)

SYSTEM_PROMPT_2026 = (
    SYSTEM_PROMPT_PAPER + "\n\n" + CLAUSULA_TAXATIVA + "\n\n" + _SEGUNDO_EJEMPLO
)

RAW_TEMPLATE = '{system}\n\nComando: "{comando}"\nJSON:'
```

- [ ] Correr y confirmar **verde** la primera mitad: `pytest -q tests/test_prompt_2026.py`

### Tarea 2 — Despacho por capacidad y construcción de entrada (TDD)

- [ ] Agregar al final de `tests/test_prompt_2026.py` los dobles de tokenizer y los tests de despacho:

```python
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

    def __call__(self, texto, **kwargs):  # no debe usarse en modo chat
        raise AssertionError("no se debe tokenizar en crudo un modelo con chat_template")


class TokenizerBase:
    """Doble mínimo de un modelo base: chat_template es None."""

    chat_template = None

    def __init__(self):
        self.texto_recibido = None

    def apply_chat_template(self, mensajes, **kwargs):  # no debe usarse
        raise AssertionError("no se debe usar chat_template en un modelo base")

    def __call__(self, texto, **kwargs):
        self.texto_recibido = texto
        return {"input_ids": [[4, 5]]}


def test_detectar_modo_con_chat_template():
    assert detectar_modo(TokenizerConChat()) == "chat_template"


def test_detectar_modo_base_por_none():
    assert detectar_modo(TokenizerBase()) == "raw_completion"


def test_detectar_modo_base_por_cadena_vacia():
    tok = TokenizerBase()
    tok.chat_template = ""
    assert detectar_modo(tok) == "raw_completion"


def test_detectar_modo_base_por_atributo_ausente():
    class SinAtributo:
        pass

    assert detectar_modo(SinAtributo()) == "raw_completion"


def test_construir_entrada_modo_chat_arma_system_y_user():
    tok = TokenizerConChat()
    entrada, modo = construir_entrada(tok, "Prendé la luz del living.")
    assert modo == "chat_template"
    assert "input_ids" in entrada
    assert tok.mensajes_recibidos[0]["role"] == "system"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_2026
    assert tok.mensajes_recibidos[1]["role"] == "user"
    assert tok.mensajes_recibidos[1]["content"] == 'Comando: "Prendé la luz del living."'
    # return_dict=True es obligatorio: sin él generate() falla con AttributeError
    assert tok.kwargs_recibidos["return_dict"] is True
    assert tok.kwargs_recibidos["add_generation_prompt"] is True
    assert tok.kwargs_recibidos["return_tensors"] == "pt"


def test_construir_entrada_modo_raw_usa_la_plantilla():
    tok = TokenizerBase()
    entrada, modo = construir_entrada(tok, "Apagá la tele.")
    assert modo == "raw_completion"
    assert "input_ids" in entrada
    assert tok.texto_recibido.startswith(SYSTEM_PROMPT_2026)
    assert 'Comando: "Apagá la tele."' in tok.texto_recibido
    assert tok.texto_recibido.rstrip().endswith("JSON:")


def test_construir_entrada_rechaza_comando_vacio():
    with pytest.raises(ValueError, match="comando vacío"):
        construir_entrada(TokenizerConChat(), "   ")
```

- [ ] Correr y confirmar que **falla** (`ImportError` de `detectar_modo`/`construir_entrada`):
  `pytest -q tests/test_prompt_2026.py`
- [ ] Implementar la segunda mitad de `src/prompt_2026.py`:

```python
def detectar_modo(tokenizer) -> ModoPrompting:
    """Decide el modo por capacidad, nunca por identificador de modelo.

    Los modelos base del roster (LFM2.5-230M / -350M) no traen chat_template;
    hardcodear sus IDs haría que el harness se rompa en silencio si mañana
    Liquid publica una variante instruct con el mismo nombre."""
    plantilla = getattr(tokenizer, "chat_template", None)
    return "raw_completion" if not plantilla else "chat_template"


def construir_entrada(tokenizer, comando: str) -> tuple[dict, ModoPrompting]:
    """Devuelve (entrada, modo), con entrada apta para modelo.generate(**entrada)."""
    if not comando or not comando.strip():
        raise ValueError(f"Se recibió un comando vacío: {comando!r}")

    modo = detectar_modo(tokenizer)
    if modo == "chat_template":
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT_2026},
            {"role": "user", "content": construir_prompt_usuario(comando)},
        ]
        entrada = tokenizer.apply_chat_template(
            mensajes,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        texto = RAW_TEMPLATE.format(system=SYSTEM_PROMPT_2026, comando=comando)
        entrada = tokenizer(texto, return_tensors="pt")
    return entrada, modo
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_prompt_2026.py`

### Tarea 3 — commit

- [ ] `pytest -q`
- [ ] `git add src/prompt_2026.py tests/test_prompt_2026.py`
- [ ] `git commit -m "feat(2026): prompt taxativo de campos categoricos y ruta de raw completion"`

## Verify

```bash
# 1. Suite completa verde
pytest -q

# 2. El prompt del paper queda incluido literalmente y el 2026 lo extiende
python -c "
import sys; sys.path.insert(0, 'src')
from prompt import SYSTEM_PROMPT_PAPER
from prompt_2026 import SYSTEM_PROMPT_2026, CLAUSULA_TAXATIVA
assert SYSTEM_PROMPT_2026.startswith(SYSTEM_PROMPT_PAPER)
assert CLAUSULA_TAXATIVA in SYSTEM_PROMPT_2026
t = CLAUSULA_TAXATIVA.lower()
assert 'unicos outputs aceptados' in t and 'violacion del formato' in t and 'sinonimo' in t
print('prompt 2026 OK,', len(SYSTEM_PROMPT_2026), 'caracteres')
"

# 3. src/prompt.py sigue byte-idéntico (F0)
git diff --exit-code main -- src/prompt.py && echo "prompt.py intacto"

# 4. El despacho no menciona ningún ID de modelo
grep -nE "LFM2\.5|LiquidAI|granite|Qwen|SmolLM2|OLMo" src/prompt_2026.py \
  && echo "FALLA: hay IDs hardcodeados" || echo "sin IDs hardcodeados OK"
```

## Acceptance criteria

- **Dado** `SYSTEM_PROMPT_2026`, **cuando** se compara con `SYSTEM_PROMPT_PAPER` importado de `src/prompt.py`, **entonces** empieza exactamente con él y lo extiende; `src/prompt.py` no fue modificado.
- **Dado** `CLAUSULA_TAXATIVA`, **entonces** nombra los cuatro campos categóricos (`intent`, `dispositivo`, `ubicacion`, `unidad`), declara que los valores entre paréntesis son los "unicos outputs aceptados" **textualmente**, califica el uso de sinónimos como "violacion del formato", y da al menos un ejemplo concreto (`televisor`/`tele` vs `tv`).
- **Dado** un tokenizer con `chat_template` no vacío, **cuando** se llama `construir_entrada`, **entonces** `detectar_modo` devuelve `"chat_template"`, se invoca `apply_chat_template` con `add_generation_prompt=True`, `return_tensors="pt"` y **`return_dict=True`**, y los mensajes son exactamente `[system=SYSTEM_PROMPT_2026, user='Comando: "..."']`.
- **Dado** un tokenizer cuyo `chat_template` es `None`, cadena vacía, o cuyo atributo no existe, **cuando** se llama `construir_entrada`, **entonces** el modo es `"raw_completion"`, no se llama `apply_chat_template`, y el texto tokenizado empieza con `SYSTEM_PROMPT_2026`, contiene `Comando: "<comando>"` y termina en `JSON:`.
- **Dado** un comando vacío o solo espacios, **cuando** se llama `construir_entrada`, **entonces** lanza `ValueError` mencionando "comando vacío".
- **Dado** `src/prompt_2026.py`, **cuando** se lo inspecciona, **entonces** no contiene ningún identificador de modelo del roster: el despacho es puramente por capacidad.
- **Dado** `pytest -q`, **entonces** pasa completo, incluida la regresión legacy.
