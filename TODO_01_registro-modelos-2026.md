---
id: 01
title: Registro de modelos 2026 y vocabularios cerrados
depends_on: []
files:
  - src/models_2026.py
  - src/taxonomia_2026.py
  - tests/test_models_2026.py
  - tests/test_taxonomia_2026.py
  - .gitignore
---

## Spec

Crear los dos módulos de datos maestros de los que depende todo el resto del DAG: el registro de los 14 modelos (F3) y los vocabularios cerrados de etiquetas y categorías (F4). Ambos son puro dato + validación, sin dependencias pesadas (nada de `torch`/`transformers`), para que se puedan importar y testear en cualquier contexto, incluido dentro de los contenedores mínimos del subtask 04.

Honra los contratos **F1** (convenciones), **F3** (registro) y **F4** (vocabularios) del índice. El roster es exactamente el de §2.1: 14 modelos, sin `Qwen2.5-*` (descartados, superados por el par Qwen3.5). `gemma-3-270m-it` y `Llama-3.2-1B-Instruct` **se mantienen**: son *gated* (requieren licencia aceptada y `$HF_TOKEN` para descargarse, ver F11), y el registro los marca con el flag `gated: bool` — sin leer ni mencionar el valor del token. Todos arrancan con `transformers_pin = BASELINE_TRANSFORMERS`, `trust_remote_code = False`, `motivo_pin = ""`; solo el subtask 04 puede cambiarlos.

## Implementation plan

### Tarea 1 — `src/taxonomia_2026.py` con parseo validado (TDD)

- [ ] Escribir el test que falla, `tests/test_taxonomia_2026.py`:

```python
"""Tests de los vocabularios cerrados de la etapa 2 (F4 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taxonomia_2026 import (  # noqa: E402
    CATEGORIAS_DISPLAY,
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_EQUIVALENTES,
    ETIQUETAS_ERROR,
    ETIQUETAS_ERROR_DISPLAY,
    parsear_categoria,
    parsear_etiquetas,
)


def test_vocabularios_tienen_el_tamano_congelado():
    assert len(ETIQUETAS_ERROR) == 7
    assert len(CATEGORIAS_LINGUISTICAS) == 6
    assert ETIQUETAS_EQUIVALENTES == {"sin_error_semantico", "uso_de_sinonimos"}


def test_las_cinco_primeras_etiquetas_son_la_tabla4_publicada():
    assert ETIQUETAS_ERROR[:5] == [
        "confusion_intencion",
        "confusion_dispositivo",
        "confusion_ubicacion",
        "alucinacion_valor_unidad",
        "valor_numerico_incorrecto",
    ]


def test_todo_valor_tiene_nombre_para_mostrar():
    assert set(ETIQUETAS_ERROR_DISPLAY) == set(ETIQUETAS_ERROR)
    assert set(CATEGORIAS_DISPLAY) == set(CATEGORIAS_LINGUISTICAS)


def test_parsear_etiquetas_acepta_lista_separada_por_punto_y_coma():
    assert parsear_etiquetas("confusion_intencion;uso_de_sinonimos") == [
        "confusion_intencion",
        "uso_de_sinonimos",
    ]


def test_parsear_etiquetas_normaliza_espacios_mayusculas_y_duplicados():
    assert parsear_etiquetas("  Confusion_Intencion ; confusion_intencion ") == [
        "confusion_intencion"
    ]


def test_parsear_etiquetas_rechaza_vacio():
    with pytest.raises(ValueError, match="al menos una etiqueta"):
        parsear_etiquetas("   ")


def test_parsear_etiquetas_rechaza_fuera_de_vocabulario_y_nombra_al_ofensor():
    with pytest.raises(ValueError, match="error_raro"):
        parsear_etiquetas("confusion_intencion;error_raro")


def test_parsear_categoria_acepta_una_sola():
    assert parsear_categoria(" Consulta_De_Estado ") == "consulta_de_estado"


def test_parsear_categoria_rechaza_dos():
    with pytest.raises(ValueError, match="exactamente una"):
        parsear_categoria("consulta_de_estado;multiples_dispositivos")


def test_parsear_categoria_rechaza_fuera_de_vocabulario():
    with pytest.raises(ValueError, match="otra_cosa"):
        parsear_categoria("otra_cosa")
```

- [ ] Correr y confirmar que **falla** por `ModuleNotFoundError: No module named 'taxonomia_2026'`:
  `pytest -q tests/test_taxonomia_2026.py`
- [ ] Implementar `src/taxonomia_2026.py` con las constantes literales de F4 y:

```python
def parsear_etiquetas(texto: str) -> list[str]:
    """Parsea la salida del juez contra el vocabulario cerrado ETIQUETAS_ERROR.

    Normaliza a minúsculas y descarta duplicados preservando el orden de
    aparición. Es deliberadamente estricto: cualquier etiqueta fuera del
    vocabulario es un error, no un valor a ignorar en silencio."""
    crudas = [p.strip().lower() for p in texto.split(SEP_ETIQUETAS)]
    crudas = [p for p in crudas if p]
    if not crudas:
        raise ValueError(f"Se esperaba al menos una etiqueta, se recibió: {texto!r}")

    vistas: list[str] = []
    for etiqueta in crudas:
        if etiqueta not in ETIQUETAS_ERROR:
            raise ValueError(
                f"Etiqueta fuera del vocabulario cerrado: {etiqueta!r}. "
                f"Válidas: {', '.join(ETIQUETAS_ERROR)}"
            )
        if etiqueta not in vistas:
            vistas.append(etiqueta)
    return vistas


def parsear_categoria(texto: str) -> str:
    """Parsea la salida del juez contra CATEGORIAS_LINGUISTICAS: exactamente una."""
    crudas = [p.strip().lower() for p in texto.split(SEP_ETIQUETAS) if p.strip()]
    if len(crudas) != 1:
        raise ValueError(
            f"Se esperaba exactamente una categoría, se recibieron {len(crudas)}: {texto!r}"
        )
    categoria = crudas[0]
    if categoria not in CATEGORIAS_LINGUISTICAS:
        raise ValueError(
            f"Categoría fuera del vocabulario cerrado: {categoria!r}. "
            f"Válidas: {', '.join(CATEGORIAS_LINGUISTICAS)}"
        )
    return categoria
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_taxonomia_2026.py`

### Tarea 2 — `src/models_2026.py` con el roster de 14 (TDD)

- [ ] Escribir el test que falla, `tests/test_models_2026.py`:

```python
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
```

> Nota para el implementador: `test_estado_inicial_sin_divergencias` es una foto del estado inicial. El subtask 04 **debe** relajarlo a un `xfail`/eliminarlo si descubre pines necesarios; `test_invariante_motivo_pin` en cambio es permanente y nunca se relaja.

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_models_2026.py`
- [ ] Implementar `src/models_2026.py` con la dataclass de F3 (incluye el campo `gated: bool`,
  ubicado **después** de `trust_remote_code` y **antes** de `motivo_pin`),
  `BASELINE_TRANSFORMERS = "transformers>=4.57.0,<5.0.0"`, los 14 `ModeloEvaluado2026` en el orden
  congelado de §2.1 (`gemma-3-270m-it` es el #7, cierra el bloque sub-1B; `Llama-3.2-1B-Instruct`
  es el #14, cierra el bloque 1-2B). Los dos gated arrancan igual que los demás:
  `transformers_pin = BASELINE_TRANSFORMERS`, `motivo_pin = ""`, `trust_remote_code = False` — no
  se les inventa un pin divergente. El módulo declara el flag `gated` pero **no** lee ni imprime el
  valor de ningún token, y no menciona `login` ni `huggingface-cli`. Además:

```python
def slug(nombre: str) -> str:
    """Nombre apto para archivo/tag de imagen: minúsculas, no-alfanumérico -> '-'."""
    return re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-")


def por_nombre(nombre: str) -> ModeloEvaluado2026:
    for modelo in MODELOS_2026:
        if modelo.nombre == nombre:
            return modelo
    raise ValueError(
        f"No existe el modelo {nombre!r} en el roster 2026. "
        f"Disponibles: {', '.join(m.nombre for m in MODELOS_2026)}"
    )


def por_tier(tier: Tier) -> list[ModeloEvaluado2026]:
    return [m for m in MODELOS_2026 if m.tier == tier]


def gated() -> list[ModeloEvaluado2026]:
    """Los modelos que requieren $HF_TOKEN para descargarse, en orden de roster."""
    return [m for m in MODELOS_2026 if m.gated]
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_models_2026.py`

### Tarea 3 — `.gitignore` de toda la rama: verificar y completar (no reescribir)

> Exención de TDD: edición de configuración. Este subtask es el **único dueño de
> `.gitignore`** en todo el DAG; ningún otro subtask debe editarlo (los subtasks 05 y 13 solo
> **verifican** que las entradas estén, y paran si faltan). El archivo **ya está commiteado**
> (commit `8acfefd`) con las entradas de caché HF, credenciales (`.env`, `*.env`, `.hf_token`), el
> `.docx` y la plantilla `LaTeX2e (1)/`. Esta tarea **no reescribe ni reordena** nada de lo
> existente: primero **verifica** que esas entradas sigan presentes, y **agrega solo lo que
> falte** — hoy, los auxiliares de compilación de LaTeX. En particular, no se tocan las entradas
> de credenciales (`.env`, `*.env`, `.hf_token`).

- [ ] Verificar que `.gitignore` ya contiene (sin modificarlas) las entradas commiteadas en
  `8acfefd`: caché de HF (`.hf_cache/`, `models_cache/`), credenciales (`.env`, `*.env`,
  `.hf_token`), y `paper_cacic_LNCS_word.docx` / `LaTeX2e (1)/` / `LaTeX2e (1).zip`.
  `git show 8acfefd:.gitignore` (o `Get-Content .gitignore`) para confirmar antes de tocar nada.
- [ ] Agregar al **final** de `.gitignore`, sin borrar ni reordenar lo que ya tiene, únicamente los
  auxiliares de LaTeX que faltan:

```gitignore
# Auxiliares de compilación de LaTeX
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
```

- [ ] Confirmar que las entradas de credenciales siguen intactas y en su lugar original:
  `git diff -- .gitignore` solo debe mostrar líneas **agregadas** al final, ninguna eliminada ni
  reordenada.
- [ ] Confirmar que el `.docx` sigue sin aparecer como no trackeado:
  `git status --short | grep -i docx` → sin salida
- [ ] Confirmar que no se ignoró nada que sí debe versionarse:
  `git check-ignore -v data/2026/detalle/x.csv paper/02_reescrito/main.tex` → sin coincidencias

### Tarea 4 — commit

- [ ] `pytest -q` (suite completa, incluida la legacy)
- [ ] `git add src/models_2026.py src/taxonomia_2026.py tests/test_models_2026.py tests/test_taxonomia_2026.py .gitignore`
- [ ] `git commit -m "feat(2026): registro de 14 modelos y vocabularios cerrados de la etapa 2"`

## Verify

```bash
# 1. Los tests nuevos y los legacy pasan juntos
pytest -q

# 2. El roster es el correcto y no arrastra modelos descartados
python -c "
import sys; sys.path.insert(0, 'src')
from models_2026 import MODELOS_2026, BASELINE_TRANSFORMERS, slug, gated
assert len(MODELOS_2026) == 14, len(MODELOS_2026)
repos = {m.hf_repo_id for m in MODELOS_2026}
prohibidos = {'Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct'}
assert not (repos & prohibidos), repos & prohibidos
assert len({slug(m.nombre) for m in MODELOS_2026}) == 14
gateados = {m.hf_repo_id for m in gated()}
assert gateados == {'google/gemma-3-270m-it','meta-llama/Llama-3.2-1B-Instruct'}, gateados
assert all(m.transformers_pin == BASELINE_TRANSFORMERS for m in MODELOS_2026 if m.hf_repo_id in gateados)
print('roster OK:', len(MODELOS_2026), 'modelos,', BASELINE_TRANSFORMERS)
"

# 3. Los módulos nuevos no arrastran dependencias pesadas
python -c "
import sys; sys.path.insert(0, 'src')
import models_2026, taxonomia_2026
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('sin torch/transformers OK')
"

# 4. Ningún archivo congelado (F0) fue tocado
git diff --name-only main -- src/models.py src/prompt.py src/scoring.py src/schema.py \
  src/metrics.py src/run_evaluation.py src/generate_figures.py tests/test_metricas.py data/
# (debe imprimir vacío)
```

## Acceptance criteria

- **Dado** el repositorio en la rama `feat/reescritura-experimento-2026`, **cuando** se importa `models_2026`, **entonces** `MODELOS_2026` tiene 14 elementos, 7 con `tier == "sub-1B"` y 7 con `tier == "1-2B"`, con `nombre` y `hf_repo_id` únicos.
- **Dado** el roster, **cuando** se buscan los repos de `Qwen2.5-*` descartados por la entrevista, **entonces** ninguno está presente; y **cuando** se filtran los `gated = True`, **entonces** son exactamente `google/gemma-3-270m-it` y `meta-llama/Llama-3.2-1B-Instruct`. `src/models_2026.py` declara el flag `gated` pero **no** lee ni imprime el valor de ningún token, y no menciona `login` ni `huggingface-cli`.
- **Dado** cualquier modelo del roster, **cuando** `transformers_pin == BASELINE_TRANSFORMERS`, **entonces** `motivo_pin == ""`; y cuando difiere, `motivo_pin` es no vacío. El test lo verifica para los 14, incluidos los dos `gated`.
- **Dado** `slug`, **cuando** se aplica a los 14 nombres, **entonces** produce 14 cadenas distintas, no vacías, compuestas solo de `[a-z0-9-]`, sin guiones en los extremos (`"Qwen3.5-0.8B" -> "qwen3-5-0-8b"`).
- **Dado** `parsear_etiquetas`, **cuando** recibe `"confusion_intencion;uso_de_sinonimos"`, **entonces** devuelve esa lista de 2; **cuando** recibe una cadena vacía o una etiqueta desconocida, **entonces** lanza `ValueError` cuyo mensaje incluye el valor ofensor.
- **Dado** `parsear_categoria`, **cuando** recibe exactamente una categoría válida (con espacios o mayúsculas), **entonces** la devuelve normalizada; **cuando** recibe cero o dos, **entonces** lanza `ValueError`.
- **Dado** `ETIQUETAS_ERROR`, **cuando** se toman sus 5 primeros elementos, **entonces** coinciden con las 5 categorías publicadas de la Tabla 4, con `alucinacion_valor_unidad` y `valor_numerico_incorrecto` **separadas** (a diferencia del legacy que las fusiona).
- **Dado** `pytest -q`, **cuando** se ejecuta desde la raíz, **entonces** pasa por completo, incluido el test de regresión legacy `tests/test_metricas.py`, sin modificarlo.
