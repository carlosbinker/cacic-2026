---
id: 16
title: Roster activo de 12, pins de transformers por grupo y matriz documentada
depends_on: [01, 04]
files:
  - src/models_2026.py
  - docker/build_all.py
  - docker/run_sweep.py
  - docker/README.md
  - tests/test_models_2026.py
  - tests/test_docker_matriz.py
  - tests/test_credenciales_gated.py
---

## Spec

Tarea correctiva (Delta 2026-08-03, ver §1 y §2.1 del índice). Corrige, **hacia adelante**, el
trabajo ya commiteado de los subtasks 01 (`37a1703`) y 04 (`99c13aa`, `dfb8315`, `d6c7581`) sin
editar ni un byte de esos commits. Ningún archivo de esta tarea es F0.

Tres cosas, todas verificadas empíricamente en Phase 3 y registradas en
`.claude-scratch/logs/probe.log`, `.claude-scratch/logs/probe5x.log` y
`.claude-scratch/logs/sweep2.log` (estos logs de sondeo **no** son datos de latencia y no se citan
como tales en ningún lado):

1. **El roster activo del barrido pasa a ser 12, no 14.** `google/gemma-3-270m-it` y
   `meta-llama/Llama-3.2-1B-Instruct` permanecen en el **registro** `MODELOS_2026` (con `gated=True`)
   pero salen del **roster activo** (`activo=False`) porque el acceso de descarga no fue otorgado:
   `403` en `/<id>/resolve/main/config.json` con un `$HF_TOKEN` válido, repos marcados
   `gated: "manual"`, aprobación pendiente al 2026-08-03. Reactivarlos en el futuro debe ser
   `activo=True` + vaciar `motivo_exclusion`, nunca una reescritura de código.
2. **La premisa de `d6c7581` (una única versión mayor de `transformers` cubre todo el roster) es
   falsa.** `granite-4.0-350m` falla bajo `transformers` 5.14.1 y `LFM2.5-230M`, `LFM2.5-350M`,
   `Qwen3.5-0.8B`, `Qwen3.5-2B` fallan bajo 4.57.6. Se vuelve a un pin **por modelo**, agrupado en
   dos grupos de versión mutuamente excluyentes: grupo A (`BASELINE_TRANSFORMERS`, resuelve 4.57.6,
   8 modelos) y grupo B (`TRANSFORMERS_5X`, resuelve 5.14.1, 4 modelos). El confusor "versiones de
   librería divergentes" se declara, no se descarta.
3. **La matriz de versiones y la tabla de exclusiones quedan documentadas** en `docker/README.md`,
   con la misma información que la futura `tabla5_versiones.tex` del subtask 11 (columna
   `¿necesario?` incluida).

**Bloqueo de infraestructura (2026-08-03).** El daemon de Docker dejó su almacenamiento en modo
solo lectura (`mkdir .../overlay2/...-init: read-only file system`,
`write .../buildkit/snapshots.db: read-only file system`): ningún `docker build`/`docker run` puede
correr mientras persista. Esto bloquea **únicamente** la Tarea 6 (reconstrucción de las 4 imágenes
del grupo B). Las Tareas 1–5 y 7 — código de `src/models_2026.py`, `docker/build_all.py`,
`docker/run_sweep.py`, `docker/README.md`, los tests nuevos y `pytest -q` — no dependen de Docker y
pueden completarse ahora.

Diseña estrictamente contra los contratos re-congelados del índice: **F3** (`src/models_2026.py`,
§4), **F5** (invariante versión↔CSV), **F8** (`tabla5_versiones.tex` desde `roster_activo()`) y
**F11** (Docker, selección por defecto = `roster_activo()`).

**Fuera de alcance, explícitamente:**
- Tocar `src/prompt_2026.py` o la ruta de *raw completion*: se conserva intacta como fallback por
  capacidad (F2), cubierta por `tests/test_prompt_2026.py` (ya existe, ya verde). RF2 ya declara que
  ningún modelo del roster activo la ejercita.
- Arrancar el barrido (`docker/run_sweep.py` sin `--dry-run`) o construir las 12 imágenes reales del
  roster completo: la única construcción real de esta tarea son las **4 imágenes del grupo B**
  (Tarea 5), para dejar el pin nuevo verificado en al menos un modelo por grupo.
- Cualquier archivo de F0.
- `data/2026/detalle/granite-4-0-350m.csv`: **se conserva**, no se toca ni se recomputa (4.57.6 es
  su pin definitivo, ya lo produjo bajo esa versión).

## Implementation plan

### Tarea 1 — `src/models_2026.py`: grupo B, `activo`/`motivo_exclusion`, `roster_activo()` (TDD)

- [x] Escribir los tests que fallan en `tests/test_models_2026.py`. **Reemplazar** las dos funciones
  que quedan obsoletas por el cambio de contrato (`test_estado_inicial_sin_divergencias` describía
  un estado transitorio que ya no existe; `test_siete_por_tier` asumía que `por_tier` filtraba el
  registro completo) y **agregar** las nuevas:

  ```python
  # --- reemplaza test_estado_inicial_sin_divergencias ---
  def test_los_pines_reflejan_los_dos_grupos_de_version():
      """Tras el Delta 02 ya no hay un unico pin de arranque: hay dos grupos,
      cada uno necesario para una parte del roster (F3, invariante b)."""
      pines = {m.transformers_pin for m in MODELOS_2026}
      assert pines == {BASELINE_TRANSFORMERS, TRANSFORMERS_5X}
      assert all(m.trust_remote_code is False for m in MODELOS_2026)


  # --- reemplaza test_siete_por_tier ---
  def test_seis_por_tier_en_el_roster_activo():
      """por_tier filtra el ROSTER ACTIVO (12), no el registro completo (14)."""
      assert len(por_tier("sub-1B")) == 6
      assert len(por_tier("1-2B")) == 6
      assert all(m.activo for m in por_tier("sub-1B") + por_tier("1-2B"))


  # --- reemplaza test_invariante_motivo_pin (ahora ata motivo_pin a "necesario", no a "diverge") ---
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
  ```

  Actualizar también el import del módulo bajo test:

  ```python
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
  ```

- [x] Correr y confirmar que **falla** (import error de `TRANSFORMERS_5X`/`roster_activo`, más los
  asserts de conteo): `pytest -q tests/test_models_2026.py`

- [x] Implementar en `src/models_2026.py`. Agregar la constante del grupo B junto a la existente:

  ```python
  # Grupo B (2026-08-03): 5 modelos recientes (LFM2.5-230M/350M, Qwen3.5-0.8B/2B)
  # exigen transformers >= 5.0.0; ver motivo_pin de cada uno. Resuelve, en esta
  # corrida, a transformers 5.14.1 -- que es precisamente la version que rompe
  # a granite-4.0-350m (grupo A). Los dos grupos son necesarios y mutuamente
  # excluyentes sobre el roster: no existe una unica version mayor que sirva
  # para las 12 filas activas. La premisa de d6c7581 queda refutada.
  TRANSFORMERS_5X: str = "transformers>=5.0.0"
  ```

  Agregar los dos campos nuevos al dataclass, al final:

  ```python
  @dataclass(frozen=True)
  class ModeloEvaluado2026:
      """Un modelo del roster 2026, con su plomeria de reproducibilidad."""

      nombre: str
      hf_repo_id: str
      params_b: float
      tier: Tier
      transformers_pin: str  # BASELINE_TRANSFORMERS o TRANSFORMERS_5X
      trust_remote_code: bool
      gated: bool  # requiere licencia aceptada + $HF_TOKEN para descargarse
      motivo_pin: str  # "" si el pin es heredado; el porque si el pin es NECESARIO
      activo: bool  # True <=> forma parte del roster activo del barrido
      motivo_exclusion: str  # "" si activo; el porque si no
  ```

  Editar cada una de las 14 entradas de `MODELOS_2026` agregando `activo=` y `motivo_exclusion=`, y
  para los 5 modelos de la tabla de abajo también `transformers_pin=` y `motivo_pin=`. Tabla completa
  (valores exactos a aplicar; los 9 modelos no listados quedan `activo=True, motivo_exclusion=""` y
  su `transformers_pin`/`motivo_pin` **no cambian**):

  | modelo | `transformers_pin` | `motivo_pin` | `activo` | `motivo_exclusion` |
  |---|---|---|---|---|
  | `LFM2.5-230M` | `TRANSFORMERS_5X` | ver abajo (A) | `True` | `""` |
  | `LFM2.5-350M` | `TRANSFORMERS_5X` | ver abajo (A) | `True` | `""` |
  | `Qwen3.5-0.8B` | `TRANSFORMERS_5X` | ver abajo (B) | `True` | `""` |
  | `Qwen3.5-2B` | `TRANSFORMERS_5X` | ver abajo (B) | `True` | `""` |
  | `granite-4.0-350m` | `BASELINE_TRANSFORMERS` (sin cambio) | ver abajo (C) | `True` | `""` |
  | `gemma-3-270m-it` | `BASELINE_TRANSFORMERS` (sin cambio) | `""` (sin cambio) | `False` | ver abajo (D) |
  | `Llama-3.2-1B-Instruct` | `BASELINE_TRANSFORMERS` (sin cambio) | `""` (sin cambio) | `False` | ver abajo (E) |

  Textos exactos de motivo (constantes de módulo, para no repetir strings largos inline y para que
  `docker/README.md` pueda citar el mismo texto con `[:N]`):

  ```python
  _MOTIVO_PIN_LFM25_BASE = (
      "Necesario: falla bajo transformers 4.57.6 con ValueError 'Tokenizer class "
      "TokenizersBackend does not exist or is not currently imported'. "
      "PROBE_OK en 5.14.1 (.claude-scratch/logs/probe5x.log)."
  )
  _MOTIVO_PIN_QWEN35 = (
      "Necesario: falla bajo transformers 4.57.6 con ValueError \"You can update "
      "Transformers with the command 'pip install --upgrade transformers'...\" "
      "(.claude-scratch/logs/probe.log)."
  )
  _MOTIVO_PIN_GRANITE_350M = (
      "Necesario: falla bajo transformers 5.14.1 con ValueError 'has_previous_state "
      "can only be called on LinearAttention layers, and the current Cache seem to "
      "only contain Attention layers' (regresion de transformers 5.x en el cache "
      "hibrido de Granite 4; .claude-scratch/logs/sweep2.log). PROBE_OK en 4.57.6 "
      "(.claude-scratch/logs/probe.log)."
  )
  _MOTIVO_EXCLUSION_GATED = (
      "Acceso de descarga no otorgado: 403 en /{repo}/resolve/main/config.json con "
      "$HF_TOKEN valido (repo gated:manual, aprobacion pendiente al 2026-08-03). "
      "Reactivar es poner activo=True y vaciar este campo, sin reescribir codigo."
  )
  ```

  Ejemplo de las 3 formas de entrada que cambian (el resto de las 14 sigue igual salvo agregar
  `activo=True, motivo_exclusion=""`):

  ```python
  ModeloEvaluado2026(
      nombre="LFM2.5-230M",
      hf_repo_id="LiquidAI/LFM2.5-230M",
      params_b=0.23,
      tier="sub-1B",
      transformers_pin=TRANSFORMERS_5X,
      trust_remote_code=False,
      gated=False,
      motivo_pin=_MOTIVO_PIN_LFM25_BASE,
      activo=True,
      motivo_exclusion="",
  ),
  # ...
  ModeloEvaluado2026(
      nombre="granite-4.0-350m",
      hf_repo_id="ibm-granite/granite-4.0-350m",
      params_b=0.35,
      tier="sub-1B",
      transformers_pin=BASELINE_TRANSFORMERS,
      trust_remote_code=False,
      gated=False,
      motivo_pin=_MOTIVO_PIN_GRANITE_350M,
      activo=True,
      motivo_exclusion="",
  ),
  # ...
  ModeloEvaluado2026(
      nombre="gemma-3-270m-it",
      hf_repo_id="google/gemma-3-270m-it",
      params_b=0.27,
      tier="sub-1B",
      transformers_pin=BASELINE_TRANSFORMERS,
      trust_remote_code=False,
      gated=True,
      motivo_pin="",
      activo=False,
      motivo_exclusion=_MOTIVO_EXCLUSION_GATED.format(repo="google/gemma-3-270m-it"),
  ),
  ```

  Agregar la función y actualizar `por_tier`:

  ```python
  def roster_activo() -> list[ModeloEvaluado2026]:
      """Los 12 modelos con activo=True, en orden de registro."""
      return [m for m in MODELOS_2026 if m.activo]


  def por_tier(tier: Tier) -> list[ModeloEvaluado2026]:
      """Filtra el ROSTER ACTIVO (no el registro completo): F3 lo re-congela asi."""
      return [m for m in roster_activo() if m.tier == tier]
  ```

  `por_nombre` **no cambia**: sigue buscando en `MODELOS_2026` completo (para poder reactivar un
  excluido hay que poder encontrarlo). `gated()` **no cambia**: sigue devolviendo los 2 del registro.

  Reescribir el docstring/comentario de módulo y el comentario de `BASELINE_TRANSFORMERS` para que
  digan lo que dice la nota de versiones re-congelada del índice (§4, F3): que no hay una única
  versión mayor que cubra el roster, que 4.57.x y 5.x son ambas necesarias y mutuamente excluyentes,
  y que el pin es por modelo, agrupado en dos grupos, con la matriz de `docker/README.md` como fuente
  única.

- [x] Correr y confirmar **verde**: `pytest -q tests/test_models_2026.py`
- [x] `pytest -q` (suite completa; confirma que nada de F0 ni de los subtasks previos se rompió)
- [x] `git add src/models_2026.py tests/test_models_2026.py` *(consolidado en el commit único final, ver nota de Cierre)*
- [x] `git commit -m "fix(2026): roster activo de 12 y pins de transformers por grupo de version"` *(consolidado: el commit real de esta corrida es único, ver Cierre)*

### Tarea 2 — `docker/build_all.py` y `docker/run_sweep.py`: seleccion por defecto = roster activo (TDD)

> Nota de mapeo de archivos: la orquestacion Docker (`comando_build`, `comando_run`,
> `seleccionar_modelos` de `docker/build_all.py` y `docker/run_sweep.py`) ya se testea en
> `tests/test_docker_matriz.py`, no en `tests/test_run_sweep_2026.py` (ese archivo cubre
> `src/run_sweep_2026.py`, el harness que corre DENTRO de cada contenedor, un modulo distinto). Los
> tests nuevos de esta tarea van en `tests/test_docker_matriz.py` para no duplicar la logica de carga
> de `docker/` por ruta que ese archivo ya resuelve.

- [x] Escribir los tests que fallan, agregar a `tests/test_docker_matriz.py`:

  ```python
  from models_2026 import roster_activo  # noqa: E402  (agregar al import existente)


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
  ```

  Agregar `import pytest` al encabezado de `tests/test_docker_matriz.py` si no está ya.

- [x] Correr y confirmar que **falla**: `pytest -q tests/test_docker_matriz.py`

- [x] Implementar en `docker/build_all.py` (reemplaza la función existente):

  ```python
  from models_2026 import MODELOS_2026, ModeloEvaluado2026, por_nombre, roster_activo, slug  # noqa: E402

  def seleccionar_modelos(nombre: str | None) -> list[ModeloEvaluado2026]:
      """El roster ACTIVO (12), o solo el modelo pedido si esta activo."""
      if nombre is None:
          return roster_activo()
      modelo = por_nombre(nombre)
      if not modelo.activo:
          raise ValueError(
              f"{nombre!r} esta excluido del roster activo: {modelo.motivo_exclusion} "
              f"Reactivarlo requiere poner activo=True en src/models_2026.py."
          )
      return [modelo]
  ```

  Actualizar el `--help` del parser (`"construye una imagen por modelo del roster 2026"` →
  `"...del roster ACTIVO (12) del 2026"`) y el docstring de módulo, que hoy dice "una imagen por
  modelo del roster" sin distinguir activo/registro.

- [x] Implementar en `docker/run_sweep.py` (reemplaza `seleccionar_modelos`, que hoy solo maneja
  `--desde` sobre `MODELOS_2026`):

  ```python
  from models_2026 import ModeloEvaluado2026, por_nombre, roster_activo  # noqa: E402  (ajustar import existente)

  def seleccionar_modelos(desde: str | None) -> list[ModeloEvaluado2026]:
      """El roster ACTIVO (12) desde el modelo indicado en adelante (RF3)."""
      modelos = roster_activo()
      if not desde:
          return modelos
      modelo_desde = por_nombre(desde)
      if not modelo_desde.activo:
          raise ValueError(
              f"{desde!r} esta excluido del roster activo: {modelo_desde.motivo_exclusion} "
              f"No se puede retomar un barrido desde un modelo que no corre."
          )
      nombres = [m.nombre for m in modelos]
      return modelos[nombres.index(desde):]
  ```

  Actualizar el docstring de módulo (hoy dice "las 14 imágenes" en dos lugares) y el `description=`
  del parser ("Barrido 2026 completo: las 14 imágenes, de a una" → "...las 12 imágenes del roster
  activo, de a una").

- [x] Correr y confirmar **verde**: `pytest -q tests/test_docker_matriz.py`
- [x] Actualizar los tests existentes que asumían 14 en `tests/test_docker_matriz.py` y
  `tests/test_run_sweep_2026.py` allí donde iteran `MODELOS_2026` para ejercitar `comando_build`/
  `comando_run` (esos tests siguen siendo válidos: `comando_build`/`comando_run` operan sobre
  cualquier `ModeloEvaluado2026` que reciban, activo o no; no hace falta tocarlos). Confirmar
  corriendo la suite completa: `pytest -q` *(verificado: `test_run_sweep_2026.py` no referencia
  `MODELOS_2026` ni conteos, no requirió cambios)*
- [x] `git add docker/build_all.py docker/run_sweep.py tests/test_docker_matriz.py` *(consolidado en el commit único final)*
- [x] `git commit -m "fix(2026): docker build/run seleccionan el roster activo por defecto"` *(consolidado, ver Cierre)*

### Tarea 3 — `docker/README.md`: matriz de versiones y tabla de exclusiones

> Exención de TDD: es documentación. Su exactitud la verifica `tests/test_docker_matriz.py`
> (Tarea 4) contra `src/models_2026.py`, no un test de esta tarea.

- [x] Reemplazar la sección `## Matriz de versiones` completa por la de los 12 modelos del roster
  activo, agrupada por grupo de versión, con la columna `¿necesario?`:

  ```markdown
  ## Matriz de versiones

  Dos grupos de version, no doce pines distintos. Evidencia por modelo, nunca inferida por familia
  (ver `.claude-scratch/logs/probe.log`, `.claude-scratch/logs/probe5x.log`,
  `.claude-scratch/logs/sweep2.log` -- sondeos de compatibilidad, nunca datos de latencia).

  ### Grupo A -- `BASELINE_TRANSFORMERS = "transformers>=4.57.0,<5.0.0"`, resuelve 4.57.6 (8 modelos)

  | Modelo | ¿Necesario? | Evidencia |
  |---|---|---|
  | `granite-4.0-350m` | **si** | Falla bajo 5.14.1: `ValueError: has_previous_state can only be called on LinearAttention layers, and the current Cache seem to only contain Attention layers` (regresion de transformers 5.x en el cache hibrido de Granite 4). `PROBE_OK` en 4.57.6. |
  | `granite-4.0-h-350m` | heredado | `PROBE_OK` en 4.57.6; 5.x no probado. |
  | `granite-4.0-1b` | heredado | idem. |
  | `granite-4.0-h-1b` | heredado | idem. |
  | `SmolLM2-360M-Instruct` | heredado | idem. |
  | `SmolLM2-1.7B-Instruct` | heredado | idem. |
  | `LFM2.5-1.2B-Instruct` | heredado | idem. |
  | `OLMo-2-0425-1B-Instruct` | heredado | idem. |

  ### Grupo B -- `TRANSFORMERS_5X = "transformers>=5.0.0"`, resuelve 5.14.1 (4 modelos)

  | Modelo | ¿Necesario? | Evidencia |
  |---|---|---|
  | `LFM2.5-230M` | **si** | Falla bajo 4.57.6: `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported`. `PROBE_OK` en 5.14.1. |
  | `LFM2.5-350M` | **si** | idem (mismo error en 4.57.6, `PROBE_OK` en 5.14.1). |
  | `Qwen3.5-0.8B` | **si** | Falla bajo 4.57.6: `ValueError: You can update Transformers with the command 'pip install --upgrade transformers'...` |
  | `Qwen3.5-2B` | **si** | idem (mismo error bajo 4.57.6). |

  **La evidencia clave.** El grupo B resuelve a `transformers 5.14.1`, que es precisamente la version
  que rompe a `granite-4.0-350m` (grupo A). Es la prueba mas limpia posible de que ninguna version
  mayor unica cubre el roster: 4.57.6 y 5.14.1 son ambas necesarias y mutuamente excluyentes. La
  premisa del commit `d6c7581` (que una unica version mayor elimina el confusor) queda refutada por
  esto, y el confusor "versiones de libreria divergentes entre modelos" se declara en metodologia y
  en Amenazas a la Validez del paper, no se descarta.

  ## Exclusiones del roster activo

  | Modelo | Endpoint que devolvio 403 | Fecha |
  |---|---|---|
  | `google/gemma-3-270m-it` | `/google/gemma-3-270m-it/resolve/main/config.json` | 2026-08-03 |
  | `meta-llama/Llama-3.2-1B-Instruct` | `/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json` | 2026-08-03 |

  Los dos repos estan marcados `gated: "manual"`: el token es valido (verificado con `whoami-v2` y
  con `/api/models/<id>`, ambos 200), pero el unico endpoint que prueba acceso de DESCARGA es
  `/<id>/resolve/<rev>/<archivo>`, y ese devuelve 403 para los dos. Permanecen en `MODELOS_2026`
  (`gated=True`, `activo=False`) para que reactivarlos, si la aprobacion llega, sea un cambio de flag
  y no una reescritura de codigo.
  ```

- [x] Reemplazar el párrafo de "Re-congelamiento del baseline (2026-08-02)" por una remisión corta a
  la nota de versiones re-congelada del índice (evitar duplicar la prosa larga en dos lugares):
  agregar una línea `Ver la nota de versiones de F3 en el índice (TODO.md §4) para la narrativa
  completa; esta matriz es la fuente de datos, esa nota es la fuente de la interpretación.`
- [x] En la sección `## Uso`, cambiar `python docker/build_all.py            # construye las 14
  imágenes` por `# construye las 12 imágenes del roster activo`, y agregar una línea documentando
  `--modelo <excluido>` fallando con el motivo.
- [x] En "Credenciales de los modelos gated", agregar una frase: *el roster activo tiene cero modelos
  gated; esta sección describe la plomería que se conserva por si se reactiva alguno de los dos
  excluidos.*
- [x] `git add docker/README.md` *(consolidado en el commit único final)*
- [x] `git commit -m "docs(docker): matriz de versiones por grupo y tabla de exclusiones del roster"` *(consolidado, ver Cierre)*

### Tarea 4 — Test de coherencia README ↔ registro (TDD)

- [x] Escribir el test que falla, agregar a `tests/test_docker_matriz.py`:

  ```python
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
  ```

- [x] Correr y confirmar **verde**: `pytest -q tests/test_docker_matriz.py`
- [x] `pytest -q` (suite completa)
- [x] `git add tests/test_docker_matriz.py` *(consolidado en el commit único final)*
- [x] `git commit -m "test(2026): coherencia README-registro de la matriz de versiones y exclusiones"` *(consolidado, ver Cierre)*

### Tarea 5 — `tests/test_credenciales_gated.py`: roster activo sin credenciales (TDD)

> El índice (`TODO.md`, bloque G1) ya documenta el contenido exacto de estas dos pruebas nuevas;
> agregarlas acá es aplicar esa documentación al archivo real por primera vez.

- [x] Escribir los tests que fallan, agregar a `tests/test_credenciales_gated.py` (agregar
  `roster_activo` al import existente `from models_2026 import MODELOS_2026, gated, por_nombre`):

  ```python
  def test_el_roster_activo_no_tiene_ningun_modelo_gated():
      assert len(roster_activo()) == 12
      assert all(not m.gated for m in roster_activo())


  def test_el_roster_activo_corre_sin_credenciales_aunque_falte_env(run_sweep, tmp_path):
      # tmp_path no tiene .env: si roster_activo() necesitara alguna credencial, esto abortaria.
      assert run_sweep.validar_credenciales(roster_activo(), tmp_path) is None
  ```

- [x] Correr y confirmar **verde**: `pytest -q tests/test_credenciales_gated.py`
- [x] `pytest -q` (suite completa)
- [x] `git add tests/test_credenciales_gated.py` *(consolidado en el commit único final)*
- [x] `git commit -m "test(2026): el roster activo de 12 corre sin \$HF_TOKEN"` *(consolidado, ver Cierre)*

### Tarea 6 — Cola de imágenes Docker: **TRASLADADA al subtask 17** (Delta 2026-08-04)

> Exención de TDD: son operaciones de build, no comportamiento nuevo.

**Estado al 2026-08-04: esta tarea ya no se ejecuta acá.** Sus tres casillas quedaron abiertas por el
bloqueo de infraestructura de Docker del 2026-08-03, y el **Delta 2026-08-04** las dejó además
**numéricamente incorrectas**: el intercambio de roster (sale `Qwen3.5-2B`, entra
`Qwen2.5-1.5B-Instruct`) cambia el reparto por grupo de versión del roster activo de **8 A / 4 B** a
**9 A / 3 B**.

Conteos corregidos, que son los que se ejecutan:

- Imágenes del grupo B a reconstruir: **3** (`LFM2.5-230M`, `LFM2.5-350M`, `Qwen3.5-0.8B`) — **no 4**.
  La de `Qwen3.5-2B` **no se construye**: ese modelo quedó excluido del roster activo, y pedirla debe
  fallar con `ValueError`.
- Imágenes del grupo A a **construir**: **1**, la de `Qwen2.5-1.5B-Instruct`
  (`slm-domotica-2026:qwen2-5-1-5b-instruct`), que **no existe** en este host.
- Imágenes del grupo A que **no** se reconstruyen: **8**, las que ya existen y cuyo pin
  (`BASELINE_TRANSFORMERS`) no cambió. 8 + 1 = **9** del grupo A, más **3** del grupo B = las **12** del
  roster activo.

**Por qué se trasladó y no se editó en su lugar.** Estas casillas dependen del roster **posterior** al
intercambio, o sea del subtask **17**; y el 17 es correctivo de este subtask, así que
`17.depends_on ∋ 16`. Dejarlas acá habría exigido `16.depends_on ∋ 17` a la vez: un **ciclo** en el DAG.
Se ejecutan en **`TODO_17_intercambio-roster-y-ejecucion.md`, Tarea 7**, con los conteos de arriba.

El núcleo de código de este subtask está **commiteado e inmutable** (`864c694`); esta nota es
documentación forward-only y no reescribe historia.

### Tarea 7 — Cierre

- [x] `pytest -q` — suite completa verde. *(108 passed, 4 skipped)*
- [x] `python docker/build_all.py --dry-run` — lista exactamente 12 líneas, cada una con el
  `TRANSFORMERS_PIN` correcto para su grupo. *(8 grupo A, 4 grupo B, verificado)*
- [x] `git grep -iE 'hf_[A-Za-z0-9]{20,}'` → vacío (exit 1).
- [x] `git diff --stat main -- data/resultados_experimento_detalle.csv
  data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv tests/test_metricas.py
  src/models.py src/prompt.py src/scoring.py src/schema.py src/metrics.py src/run_evaluation.py
  src/generate_figures.py` → vacío (F0 intacto).
- [x] Confirmar que `data/2026/detalle/granite-4-0-350m.csv` sigue byte-idéntico a como estaba antes
  de esta tarea: `git diff --stat -- data/2026/detalle/granite-4-0-350m.csv` → vacío.

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. roster_activo() tiene 12, 6 por tier, cero gated
python -c "
import sys; sys.path.insert(0,'src')
from models_2026 import roster_activo
r = roster_activo()
assert len(r) == 12
assert sum(1 for m in r if m.tier == 'sub-1B') == 6
assert sum(1 for m in r if m.tier == '1-2B') == 6
assert not any(m.gated for m in r)
print('roster_activo OK: 12, 6/6, sin gated')
"

# 3. Dos grupos de version, 8 y 4, mutuamente excluyentes
python -c "
import sys; sys.path.insert(0,'src')
from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, roster_activo
r = roster_activo()
a = [m for m in r if m.transformers_pin == BASELINE_TRANSFORMERS]
b = [m for m in r if m.transformers_pin == TRANSFORMERS_5X]
assert len(a) == 8 and len(b) == 4
print('grupos OK: A=8, B=4')
"

# 4. build_all --dry-run lista 12 imagenes con el pin correcto
python docker/build_all.py --dry-run | grep -c 'build slm-domotica-2026:'   # -> 12
python docker/build_all.py --dry-run | grep -c 'transformers>=5.0.0'        # -> 4
python docker/build_all.py --dry-run | grep -c 'transformers>=4.57.0,<5.0.0'  # -> 8

# 5. Pedir un modelo excluido falla con el motivo
python docker/build_all.py --modelo "gemma-3-270m-it" --dry-run ; echo "exit=$?"
# -> exit=1, y el mensaje de error nombra "403" o "resolve"

# 6. run_sweep --dry-run sin --desde corre 12, ninguna con --env-file
python docker/run_sweep.py --dry-run | grep -c '^==='            # -> 12
python docker/run_sweep.py --dry-run | grep -c -- '--env-file'   # -> 0

# 7. transformers efectivo por grupo -- SUPERSEDIDO por TODO_17, Tarea 7 y Verify 8.
#    Los conteos de este bloque (12 builds, 4 del grupo B, 8 del grupo A) valen para el
#    roster PREVIO al Delta 2026-08-04; el reparto vigente es 9 A / 3 B. Ver TODO_17.
docker run --rm slm-domotica-2026:lfm2-5-230m python -c "import transformers; print(transformers.__version__)"
docker run --rm slm-domotica-2026:granite-4-0-350m python -c "import transformers; print(transformers.__version__)"

# 8. Sin secretos versionados
git grep -iE 'hf_[A-Za-z0-9]{20,}' ; test $? -eq 1 && echo "sin token versionado OK"

# 9. granite-4-0-350m.csv no se toco
git diff --stat -- data/2026/detalle/granite-4-0-350m.csv   # -> vacio

# 10. F0 intacto
git diff --stat main -- data/resultados_experimento_detalle.csv \
  data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv \
  tests/test_metricas.py src/models.py src/prompt.py src/scoring.py src/schema.py \
  src/metrics.py src/run_evaluation.py src/generate_figures.py
```

## Acceptance criteria

> **Nota post-Delta 03 (2026-08-04), forward-only.** Los criterios de abajo se evaluaron y se
> cumplieron contra el roster **previo** al intercambio, y su código está commiteado en
> `864c694`. Tres de ellos quedaron numéricamente superados por el Delta 2026-08-04 y **los reemplaza
> `TODO_17_intercambio-roster-y-ejecucion.md`**, que es la única fuente vigente para esos números:
>
> - "`MODELOS_2026` sigue teniendo exactamente 14 elementos" → **15**.
> - "`transformers_pin` toma dos valores, con 8 y 4 modelos" → **9 y 3** sobre el roster activo
>   (**11 y 4** sobre el registro).
> - "`motivo_pin != ""` … ese conjunto es exactamente `{LFM2.5-230M, LFM2.5-350M, Qwen3.5-0.8B,
>   Qwen3.5-2B, granite-4.0-350m}`" → sigue siendo ese conjunto sobre el **registro**, pero sobre el
>   **roster activo** son **4** (sin `Qwen3.5-2B`, que quedó excluido conservando su `motivo_pin`).
> - "las 4 imágenes del grupo B reconstruidas" → **3** reconstrucciones del grupo B más **1**
>   construcción nueva del grupo A (ver Tarea 6 de este archivo y Tarea 7 del subtask 17).
>
> El resto de los criterios sigue vigente sin cambios. Este texto no se reescribe: se anota.

- **Dado** `src/models_2026.py`, **entonces** declara `TRANSFORMERS_5X = "transformers>=5.0.0"`,
  el dataclass `ModeloEvaluado2026` tiene los campos `activo: bool` y `motivo_exclusion: str`, y
  `MODELOS_2026` sigue teniendo exactamente 14 elementos.
- **Dado** `roster_activo()`, **entonces** devuelve exactamente 12 modelos, en el mismo orden relativo
  que `MODELOS_2026`, con 6 por tier y ninguno `gated=True`.
- **Dado** cualquier modelo de `MODELOS_2026`, **entonces** `activo == False` si y solo si
  `motivo_exclusion != ""`, y los dos únicos con `activo == False` son `google/gemma-3-270m-it` y
  `meta-llama/Llama-3.2-1B-Instruct`, cada uno con su motivo citando el 403 de `/resolve/`.
- **Dado** cualquier modelo de `MODELOS_2026`, **entonces** `motivo_pin != ""` si y solo si su pin es
  necesario (evidencia empírica de fallo con la otra versión mayor), y ese conjunto es exactamente
  `{LFM2.5-230M, LFM2.5-350M, Qwen3.5-0.8B, Qwen3.5-2B, granite-4.0-350m}`.
- **Dado** el roster activo, **entonces** `transformers_pin` toma exactamente dos valores
  (`BASELINE_TRANSFORMERS`, `TRANSFORMERS_5X`), con 8 y 4 modelos respectivamente.
- **Dado** `por_tier`, **entonces** filtra el roster activo (no el registro): `por_tier("sub-1B")` y
  `por_tier("1-2B")` devuelven 6 cada uno. `por_nombre` sigue resolviendo cualquiera de los 14.
- **Dado** `python docker/build_all.py --dry-run` (sin `--modelo`), **entonces** planea exactamente 12
  builds, con `--build-arg TRANSFORMERS_PIN=` igual al pin de cada modelo de `roster_activo()`.
- **Dado** `python docker/build_all.py --modelo "gemma-3-270m-it"` o `"Llama-3.2-1B-Instruct"`,
  **entonces** falla con `ValueError` (no construye nada) y el mensaje incluye el `motivo_exclusion`
  de ese modelo.
- **Dado** `python docker/run_sweep.py --dry-run` (sin `--desde`), **entonces** planea exactamente 12
  invocaciones y **ninguna** lleva `--env-file`.
- **Dado** `python docker/run_sweep.py --desde "gemma-3-270m-it"`, **entonces** falla con
  `ValueError` nombrando el `motivo_exclusion`, sin ejecutar ningún contenedor.
- **Dado** `docker/README.md`, **entonces** contiene la matriz de los 12 modelos del roster activo
  agrupada en grupo A / grupo B con la columna `¿necesario?`, y una tabla de exclusiones con los 2
  modelos, el endpoint 403 y la fecha `2026-08-03`.
- **Dado** las 4 imágenes del grupo B reconstruidas, **entonces** `docker run ... python -c "import
  transformers; print(transformers.__version__)"` devuelve una versión `5.*` en al menos una de
  ellas, y la imagen de `granite-4.0-350m` (grupo A) sigue devolviendo `4.57.*`.
- **Dado** `data/2026/detalle/granite-4-0-350m.csv`, **entonces** sigue byte-idéntico a como estaba
  antes de esta tarea: nadie lo recomputó.
- **Dado** el repositorio al final, **entonces** `pytest -q` pasa, ningún archivo de F0 cambió,
  `git grep -iE 'hf_[A-Za-z0-9]{20,}'` sale vacío, y `src/prompt_2026.py` no fue tocado.
