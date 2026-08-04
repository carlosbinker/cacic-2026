---
id: 17
title: "Intercambio de roster (sale Qwen3.5-2B, entra Qwen2.5-1.5B-Instruct), pinning de núcleos y continuación ante fallo"
depends_on: [01, 04, 16]
files:
  - src/models_2026.py
  - docker/run_sweep.py
  - docker/README.md
  - tests/test_models_2026.py
  - tests/test_docker_matriz.py
  - tests/test_credenciales_gated.py
---

## Spec

Tarea correctiva del **Delta 2026-08-04** (ver §1 `### Delta 2026-08-04`, §2.1, y los contratos F3,
F5, F8, F11 y F12 del índice). Corrige **hacia adelante** el trabajo ya commiteado del subtask 16
(`864c694`) sin editar ni un byte de ese commit ni de su texto. Ningún archivo de esta tarea es F0.

Tres cambios, todos decididos por el usuario y ninguno inventado acá:

1. **Intercambio de roster, a conteo constante.** `Qwen3.5-2B` sale del roster activo **sin
   condición** y queda en el registro con `activo=False`. `Qwen/Qwen2.5-1.5B-Instruct` entra al roster
   activo, grupo de versión **A**, con evidencia de sonda propia. El **roster activo sigue en 12**, con
   **6 por tier**; el **registro pasa de 14 a 15** y los **excluidos de 2 a 3**; el reparto por grupo de
   versión sobre el roster activo pasa de **8 A / 4 B** a **9 A / 3 B**.
2. **Pinning de núcleos (F11, RNF6).** `docker run` gana `--cpuset-cpus`, con el mismo valor en las 12
   corridas del barrido y en la del juez. Sin esto, `--cpus=2` es solo una cuota de CFS y la latencia
   arrastra variabilidad por migración de núcleos.
3. **Continuación ante fallo y registro de fallos (F12.1, RF20c).** `docker/run_sweep.py` deja de
   cortar al primer fallo de modelo: registra el fallo con su error textual, sigue con el siguiente, y
   escribe `data/2026/fallos_barrido.json`. **El barrido no puede bloquearse por un modelo: el usuario
   no está disponible para desbloquearlo.** La única excepción sigue siendo AUTH-STOP (401/403 corta
   inmediato, sin reintentos, sin comandos de login).

**Por qué es una tarea nueva y no una edición del subtask 16.** Los seis archivos de esta tarea son
exactamente los que `864c694` dejó commiteados. Ese commit es **inmutable**: su texto y sus casillas
tildadas quedan como están. Todo cambio sobre ese material entra por acá.

**Por qué la cola de builds del subtask 16 se trasladó acá (Tarea 7).** Las tres casillas de Docker
que quedaban abiertas en `TODO_16` dependen del roster **posterior** al intercambio, o sea de esta
tarea. Dejarlas en 16 habría exigido `16.depends_on ∋ 17` y `17.depends_on ∋ 16` a la vez: un **ciclo**
en el DAG. Se mueven acá con los conteos corregidos.

### Evidencia del grupo de versión de `Qwen2.5-1.5B-Instruct` — grupo A, sondeado, no inferido

**Regla del método, sin excepción:** ninguna fila fija su `transformers_pin` por inferencia de familia.
Inferir por familia es exactamente el error que habría escondido el fallo de Granite 4, y la propia
matriz muestra que "familia Qwen" **no** predice versión (Qwen3.5 resuelve a grupo B).

Sonda ejecutada el 2026-08-04 dentro de la imagen ya construida del grupo A,
`slm-domotica-2026:granite-4-0-350m`, con su versión efectiva verificada en el contenedor
(`docker run --rm slm-domotica-2026:granite-4-0-350m python -c "import transformers; print(transformers.__version__)"`
→ `4.57.6`). Salida verbatim, en `.claude-scratch/logs/probe_qwen25_15b_groupA.log`:

```
PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template
```

Carga real del modelo + `apply_chat_template` funcionando + `generate()` greedy de 8 tokens.
**Fuerza de la evidencia: fuerte pero parcial**, el mismo estándar que las filas *heredadas* del grupo
A: confirma que 4.57.6 funciona; no prueba ni descarta 5.14.1 (esa sonda no se corrió). Por lo tanto su
pin es **heredado** → `motivo_pin == ""` (F3 (a) e (i)). Los tiempos de esa sonda **no** son datos de
latencia y no pueden llegar a `data/2026/**`, a ninguna tabla o figura, ni a ninguna afirmación de
latencia del paper.

### Corrección de ID, registrada para que nadie la repita

El usuario escribió *"Qwen-2.5-1.7-instruct"*. Ese modelo **no existe**:
`Qwen/Qwen2.5-1.7B-Instruct` → **HTTP 404**. El ID real es **`Qwen/Qwen2.5-1.5B-Instruct`** → HTTP 200,
y es uno de los dos Qwen2.5 del paper original, así que el usuario lo confirmó al elegirlo entre las
opciones. Un test de esta tarea fija los dos hechos (F3 (h)).

### `Qwen3.5-2B`: redacción taxativa de su exclusión

Motivo registrado: falla bajo 4.57.6 con `ValueError` que pide `pip install --upgrade transformers`
(evidencia negativa propia, `.claude-scratch/logs/probe.log`) y su **confirmación positiva bajo 5.14.1
nunca se obtuvo**, porque la sonda no llegó a correr (almacenamiento del daemon de Docker en modo solo
lectura a mitad de ronda: bloqueo de infraestructura, **no** evidencia sobre el modelo). El usuario
eligió excluirlo en lugar de perseguir esa confirmación.

**Nunca escribir que falla bajo las dos versiones mayores.** No hay evidencia de que falle bajo 5.14.1;
hay **ausencia de evidencia** de que funcione. La redacción correcta es "compatibilidad no verificada".
Conserva su `transformers_pin` del grupo B y su `motivo_pin` (su pin sigue siendo **necesario** por la
evidencia negativa bajo 4.57.6): `motivo_exclusion` y `motivo_pin` son ortogonales (F3 (g)).

**Fuera de alcance, explícitamente:**

- Sondear `Qwen3.5-2B` bajo 5.14.1. El usuario decidió excluirlo; §2.4 lo declara fuera de alcance.
- Sondear `Qwen2.5-1.5B-Instruct` bajo 5.14.1: no hace falta para fijar su grupo, y su pin es heredado.
- Arrancar el barrido (`docker/run_sweep.py` sin `--dry-run`). Eso es el subtask 05.
- Tocar `src/run_sweep_2026.py`, `src/prompt_2026.py`, `src/judge_2026.py` o `docker/build_all.py`:
  `build_all.py` ya deriva todo de `roster_activo()` y no necesita cambios; la persistencia del juez
  (F12.3) es del subtask 07.
- Cualquier archivo de F0. En particular `data/resultados_experimento_resumen.json` se **lee** (para la
  tabla de continuidad de §2.1) y jamás se escribe.
- `data/2026/detalle/granite-4-0-350m.csv`: **se conserva**, no se toca ni se recomputa.

## Implementation plan

### Tarea 1 — `src/models_2026.py`: intercambio de roster (TDD)

- [ ] Escribir los tests que fallan en `tests/test_models_2026.py`. **Reemplazar** las funciones cuyo
  conteo quedó obsoleto y **agregar** las nuevas:

  ```python
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


  def test_nombres_y_repos_son_unicos():
      assert len({m.nombre for m in MODELOS_2026}) == 15
      assert len({m.hf_repo_id for m in MODELOS_2026}) == 15


  def test_los_slugs_son_unicos_y_aptos_para_nombre_de_archivo():
      slugs = [slug(m.nombre) for m in MODELOS_2026]
      assert len(set(slugs)) == 15
      for s in slugs:
          assert s and all(c.isalnum() or c == "-" for c in s)
          assert not s.startswith("-") and not s.endswith("-")


  def test_slug_del_qwen25_reincorporado():
      assert slug("Qwen2.5-1.5B-Instruct") == "qwen2-5-1-5b-instruct"


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
      import json
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
  ```

  El archivo necesita `import json` y `from pathlib import Path` en la cabecera si todavía no los tiene.

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_models_2026.py`
  → esperado: fallan al menos `test_el_registro_tiene_quince_modelos`,
  `test_dos_grupos_de_version_nueve_y_tres_en_el_roster_activo`,
  `test_qwen25_15b_esta_activo_en_el_grupo_a_con_pin_heredado` (con `ValueError` de `por_nombre`) y
  `test_los_tres_excluidos_y_sus_dos_causas`.

- [ ] Agregar la constante de motivo de exclusión, junto a las que ya existen en
  `src/models_2026.py`:

  ```python
  _MOTIVO_EXCLUSION_QWEN35_2B = (
      "Fuera del roster activo por decision del usuario (2026-08-04): su pin del grupo B es "
      "necesario (falla bajo 4.57.6 con ValueError que pide 'pip install --upgrade "
      "transformers'), pero su confirmacion positiva bajo 5.14.1 NUNCA SE OBTUVO -- la sonda no "
      "llego a correr porque el almacenamiento del daemon de Docker quedo en modo solo lectura a "
      "mitad de ronda (bloqueo de infraestructura, NO evidencia sobre el modelo). Compatibilidad "
      "NO VERIFICADA, no incompatibilidad demostrada: no hay evidencia de que falle bajo 5.14.1. "
      "El usuario eligio excluirlo en lugar de perseguir esa confirmacion. Reactivarlo es poner "
      "activo=True, vaciar este campo y obtener la sonda positiva."
  )
  ```

- [ ] Marcar `Qwen3.5-2B` como excluido (**única** modificación de su entrada: `activo` y
  `motivo_exclusion`; `transformers_pin` y `motivo_pin` **no se tocan**):

  ```python
      ModeloEvaluado2026(
          nombre="Qwen3.5-2B",
          hf_repo_id="Qwen/Qwen3.5-2B",
          params_b=2.0,
          tier="1-2B",
          transformers_pin=TRANSFORMERS_5X,
          trust_remote_code=False,
          gated=False,
          motivo_pin=_MOTIVO_PIN_QWEN35,
          activo=False,
          motivo_exclusion=_MOTIVO_EXCLUSION_QWEN35_2B,
      ),
  ```

- [ ] Agregar `Qwen2.5-1.5B-Instruct` como **última** entrada de `MODELOS_2026` (posición 15). Va al
  final y no intercalado, para que el orden de registro de las 14 filas anteriores —y por lo tanto el
  orden de `roster_activo()`, de `resumen_2026.json` y de las columnas de las tablas 3 y 4— no cambie
  respecto de `864c694`:

  ```python
      ModeloEvaluado2026(
          nombre="Qwen2.5-1.5B-Instruct",
          hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
          params_b=1.54,
          tier="1-2B",
          transformers_pin=BASELINE_TRANSFORMERS,
          trust_remote_code=False,
          gated=False,
          # Pin HEREDADO: la sonda del 2026-08-04 confirma que 4.57.6 funciona
          # (PROBE_OK|Qwen2.5-1.5B-Instruct|4.57.6|chat_template, ver
          # .claude-scratch/logs/probe_qwen25_15b_groupA.log), pero NO se probo
          # 5.14.1, asi que no hay evidencia de necesidad. Grupo determinado por
          # sonda propia, nunca por inferencia de familia: Qwen3.5 resuelve a
          # grupo B, o sea que "familia Qwen" no predice version.
          motivo_pin="",
          activo=True,
          motivo_exclusion="",
      ),
  ```

- [ ] Actualizar el docstring de módulo de `src/models_2026.py` agregando, **sin borrar** lo que ya
  documenta de los Deltas anteriores, un punto 3 que registre: el registro pasa a 15; `Qwen3.5-2B`
  queda excluido por compatibilidad **no verificada** (no demostradamente incompatible);
  `Qwen2.5-1.5B-Instruct` entra al grupo A por sonda propia con pin heredado; el ID
  `Qwen/Qwen2.5-1.7B-Instruct` **no existe** (404) y no debe reintroducirse; y que el reingreso sostiene
  la línea de continuidad con la Tabla 2 publicada (RF19).
- [ ] Correr y confirmar **verde**: `pytest -q tests/test_models_2026.py`

### Tarea 2 — `docker/run_sweep.py`: `--cpuset-cpus` (TDD)

- [ ] Escribir los tests que fallan, en `tests/test_docker_matriz.py`:

  ```python
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
  ```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_docker_matriz.py -k cpuset`
  → esperado: `ImportError` de `CPUSET_POR_DEFECTO` y/o `TypeError` por el kwarg `cpuset`.
- [ ] Implementar en `docker/run_sweep.py`:

  ```python
  CPUSET_POR_DEFECTO = "0-1"


  def comando_run(modelo: ModeloEvaluado2026, raiz: Path, force: bool = False,
                  cpuset: str = CPUSET_POR_DEFECTO) -> list[str]:
      """El `docker run` congelado de ese modelo (F11).

      `--memory=8g --cpus=2` no son negociables (RNF1) y `--cpuset-cpus` tampoco
      (RNF6): `--cpus` es una cuota del planificador CFS y permite migracion
      entre nucleos, asi que sin pinning la latencia arrastra variabilidad que
      no es del modelo. El valor tiene que ser el MISMO en las 12 corridas y en
      la del juez; cual sea es secundario.
      """
      cmd = [
          "docker", "run", "--rm",
          "--memory=8g", "--cpus=2", f"--cpuset-cpus={cpuset}",
      ]
      if modelo.gated:
          # F11: exclusivamente los modelos gated reciben el token, y solo asi.
          cmd += ["--env-file", str(raiz / ".env")]
      cmd += [
          "-v", f"{raiz / 'data'}:/app/data",
          "-v", f"{raiz / '.hf_cache'}:/app/.hf_cache",
          tag_imagen(modelo),
          "python",
      ]
      if force:
          cmd += ["src/run_sweep_2026.py", "--force", "--modelo", modelo.nombre]
      else:
          cmd += ["src/run_sweep_2026.py", "--modelo", modelo.nombre]
      return cmd
  ```

- [ ] Agregar el flag al CLI, en `_parsear_argumentos()`:

  ```python
      parser.add_argument("--cpuset", default=CPUSET_POR_DEFECTO,
                          help="par de nucleos al que se fija cada contenedor "
                               "(--cpuset-cpus); el mismo para las 12 corridas")
  ```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_docker_matriz.py`

### Tarea 3 — `docker/run_sweep.py`: continuación ante fallo y `fallos_barrido.json` (TDD)

- [ ] Escribir los tests que fallan, en `tests/test_docker_matriz.py`:

  ```python
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
  ```

  El archivo necesita `import json` en la cabecera si todavía no lo tiene.

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_docker_matriz.py -k barrido`
  → esperado: `AttributeError` de `FALLOS_PATH` y, con el código viejo, `corridos` corta en el segundo
  modelo.
- [ ] Implementar en `docker/run_sweep.py`:

  ```python
  FALLOS_PATH = RAIZ / "data" / "2026" / "fallos_barrido.json"


  def _registrar_fallo(modelo: ModeloEvaluado2026, codigo: int, error: str) -> dict:
      """La fila de F5 que documenta un fallo irrecuperable de un modelo."""
      return {
          "modelo": modelo.nombre,
          "hf_repo_id": modelo.hf_repo_id,
          "transformers_pin": modelo.transformers_pin,
          "codigo_salida": codigo,
          "error_textual": error,
          "momento_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
      }


  def ejecutar_barrido(modelos: list[ModeloEvaluado2026], force: bool, dry_run: bool,
                       cpuset: str = CPUSET_POR_DEFECTO) -> int:
      """De a un contenedor por vez, en orden de roster; NO corta al primer fallo.

      F12.1 / RF20c: si un modelo falla de forma irrecuperable se registra el
      fallo con su error textual y se sigue con el siguiente. Bloquear el
      barrido entero por un modelo no es una opcion: el usuario no esta
      disponible para desbloquearlo, y los 11 restantes son datos que se
      perderian. Los modelos ya terminados quedan en disco y el harness los
      saltea, asi que relanzar no rehace nada (RF3).
      """
      fallos: list[dict] = []
      for i, modelo in enumerate(modelos, 1):
          codigo, error = _correr_modelo(modelo, f"{i}/{len(modelos)}", force,
                                         dry_run, cpuset)
          if codigo != 0:
              fallos.append(_registrar_fallo(modelo, codigo, error))
              print(f"FALLO {modelo.nombre} (codigo {codigo}). Se registra y se "
                    f"CONTINUA con el siguiente modelo.\n{error}", file=sys.stderr)

      if not dry_run:
          FALLOS_PATH.parent.mkdir(parents=True, exist_ok=True)
          FALLOS_PATH.write_text(
              json.dumps(fallos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
          )

      if fallos:
          print(f"\nBarrido terminado con {len(fallos)} fallo(s): "
                f"{', '.join(f['modelo'] for f in fallos)}. "
                f"Detalle en {FALLOS_PATH}. Retomar un modelo puntual con "
                f"--desde \"<nombre>\".", file=sys.stderr)
          return 1
      print("\nBarrido completo, sin fallos.")
      return 0
  ```

- [ ] Ajustar `_correr_modelo` para propagar el `cpuset` y devolver también el error textual:

  ```python
  def _correr_modelo(modelo: ModeloEvaluado2026, posicion: str, force: bool,
                     dry_run: bool, cpuset: str) -> tuple[int, str]:
      """Corre el contenedor de un modelo; devuelve (codigo de salida, stderr)."""
      cmd = comando_run(modelo, RAIZ, force, cpuset)
      print(f"\n=== [{posicion}] {modelo.nombre} ({modelo.tier}) ===")
      print(" ".join(cmd))
      if dry_run:
          return 0, ""
      completado = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
      if completado.stderr:
          print(completado.stderr, file=sys.stderr)
      return completado.returncode, completado.stderr or ""
  ```

- [ ] Pasar el `cpuset` desde `main()`: `return ejecutar_barrido(modelos, args.force, args.dry_run, args.cpuset)`
- [ ] Agregar los imports que faltan en `docker/run_sweep.py`: `import json` y
  `from datetime import datetime, timezone`.
- [ ] Actualizar el docstring de módulo de `docker/run_sweep.py`: selecciona `roster_activo()` (12) de
  un registro de **15**; **tres** modelos excluidos, dos gated y uno por compatibilidad no verificada;
  `--cpuset-cpus` con el mismo valor en las 12; y **no corta al primer fallo** (registra en
  `data/2026/fallos_barrido.json` y sigue), con la excepción AUTH-STOP.
- [ ] Correr y confirmar **verde**: `pytest -q tests/test_docker_matriz.py`

### Tarea 4 — `tests/test_credenciales_gated.py`: 13 no-gated (TDD)

> Exención de TDD: es el ajuste de un conteo en un test existente, no comportamiento nuevo. El test
> **ya está fallando** por el cambio de la Tarea 1, que es la mitad "rojo" del ciclo.

- [ ] Confirmar que **falla** ahora mismo, por el registro de 15:
  `pytest -q tests/test_credenciales_gated.py` → falla `test_los_doce_no_gated_corren_sin_credenciales`
  con `assert 13 == 12`.
- [ ] Renombrar y corregir ese test, dejando escrito **por qué** son 13:

  ```python
  def test_los_no_gated_corren_sin_credenciales(run_sweep, tmp_path):
      """13 tras el Delta 2026-08-04: los 12 del roster activo mas Qwen3.5-2B,
      que esta excluido pero NO es gated. El conteo de gated es del registro y no
      cambio (siguen siendo 2)."""
      no_gated = [m for m in MODELOS_2026 if not m.gated]
      assert len(no_gated) == 13
      assert run_sweep.validar_credenciales(no_gated, tmp_path) is None
  ```

- [ ] Agregar el test que fija la otra mitad del invariante, para que un futuro
  `activo=False` sobre un modelo gated no pase inadvertido:

  ```python
  def test_un_excluido_no_gated_no_dispara_el_fail_fast(run_sweep, tmp_path):
      """Qwen3.5-2B esta excluido, pero no necesita token: si alguien lo reactiva,
      el barrido tiene que seguir corriendo sin .env."""
      qwen = por_nombre("Qwen3.5-2B")
      assert qwen.activo is False and qwen.gated is False
      assert run_sweep.validar_credenciales([qwen], tmp_path) is None
  ```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_credenciales_gated.py`

### Tarea 5 — `docker/README.md`: matriz, exclusiones y `--cpuset-cpus`

> Exención de TDD: es documentación. Su corrección la verifican los tests de coherencia
> README ↔ registro que ya existen en `tests/test_docker_matriz.py` (Tarea 6 los ajusta).

- [ ] **Grupo A** — cambiar el encabezado a `(9 modelos)` y agregar la fila de
  `Qwen2.5-1.5B-Instruct`:

  ```markdown
  | `Qwen2.5-1.5B-Instruct` | heredado | `PROBE_OK` en 4.57.6 (`PROBE_OK\|Qwen2.5-1.5B-Instruct\|4.57.6\|chat_template`, `.claude-scratch/logs/probe_qwen25_15b_groupA.log`); 5.x no probado. Grupo determinado por sonda propia, nunca por inferencia de familia: Qwen3.5 resuelve a grupo B. |
  ```

- [ ] **Grupo B** — cambiar el encabezado a `(3 modelos activos + 1 registrado y excluido)`, dejar las
  tres filas activas, y **reemplazar** la fila de `Qwen3.5-2B` por una que diga que está excluido, sin
  perder su `transformers_pin` ni su `motivo_pin` (los tests de coherencia los buscan en el archivo):

  ```markdown
  | `Qwen3.5-2B` | **si** (registrado, **excluido del roster activo**) | Necesario: falla bajo transformers 4.57.6 con ValueError "You can update Transformers with the command 'pip install --upgrade transformers'..." (`.claude-scratch/logs/probe.log`). Su confirmacion positiva bajo 5.14.1 **nunca se obtuvo**: la sonda no llego a correr por el bloqueo de infraestructura de Docker. **Compatibilidad no verificada, no incompatibilidad demostrada.** Fuera del roster activo por decision del usuario (2026-08-04). |
  ```

- [ ] **Exclusiones** — la tabla pasa a **tres** filas, con una columna de causa, porque ya no todas
  las exclusiones son por 403:

  ```markdown
  ## Exclusiones del roster activo

  | Modelo | Causa | Detalle | Fecha |
  |---|---|---|---|
  | `google/gemma-3-270m-it` | acceso de descarga no otorgado | 403 en `/google/gemma-3-270m-it/resolve/main/config.json` con `$HF_TOKEN` valido (repo `gated: "manual"`) | 2026-08-03 |
  | `meta-llama/Llama-3.2-1B-Instruct` | acceso de descarga no otorgado | 403 en `/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json` con `$HF_TOKEN` valido (repo `gated: "manual"`) | 2026-08-03 |
  | `Qwen/Qwen3.5-2B` | compatibilidad de version **no verificada** | Falla bajo 4.57.6 (evidencia propia). Su sonda bajo 5.14.1 nunca corrio: el almacenamiento del daemon de Docker quedo en solo lectura. **No** hay evidencia de que falle bajo 5.14.1. Decision del usuario: excluirlo en lugar de perseguir la confirmacion. **No es gated**: su invocacion no lleva `--env-file`. | 2026-08-04 |
  ```

- [ ] Agregar la sección del envelope de recursos, con el valor de `--cpuset-cpus` y su razón:

  ```markdown
  ## Envelope de recursos (RNF1, RNF6)

  Toda invocacion de `docker run` de este proyecto -- las 12 del barrido y la del
  juez -- usa:

      --memory=8g --cpus=2 --cpuset-cpus=0-1

  `--cpus=2` es una **cuota** del planificador CFS: acota el tiempo de CPU pero
  permite migracion entre nucleos y no reserva nada. `--memory=8g` es un **techo**,
  no una reserva (Docker solo ofrece `--memory-reservation`, que es blanda).
  `--cpuset-cpus` **si** fija los nucleos y elimina tanto la competencia por tiempo
  de CPU como la variabilidad por migracion. Lo que ningun flag puede hacer es
  particionar la cache L3 ni el bus de memoria, que es donde la inferencia de LLM
  en CPU esta realmente limitada: por eso el barrido es **secuencial**, un modelo a
  la vez. Correr 4 modelos en paralelo habria inflado los tiempos por comando de
  forma invisible en la tabla final. Costo de la decision: ~7 h de reloj en lugar
  de ~2 h. Exactitud, taxonomia y equivalencia semantica no dependen de esto
  (temperatura 0, deterministas); solo la latencia lo hacia.

  Lo que hace comparable la latencia no es *cual* par de nucleos se use, sino que
  sea **el mismo par en las 12 corridas y en la del juez**. Para cambiarlo en otro
  host: `python docker/run_sweep.py --cpuset "2-3"`.
  ```

- [ ] Actualizar la sección **Uso**: el ejemplo `python docker/run_sweep.py --desde "Qwen3.5-2B"`
  ahora **fallaría** con `ValueError` (ese modelo está excluido). Reemplazarlo por un modelo activo y
  agregar el caso de fallo:

  ```markdown
      python docker/build_all.py            # construye las 12 imagenes del roster activo
      python docker/run_sweep.py            # corre el barrido completo, de a una, --cpuset-cpus=0-1
      python docker/run_sweep.py --desde "Qwen3.5-0.8B"   # retoma tras una interrupcion
      python docker/run_sweep.py --cpuset "2-3"           # otro par de nucleos, el mismo para las 12
      python docker/build_all.py --modelo "gemma-3-270m-it"   # falla: ValueError con el motivo_exclusion
      python docker/run_sweep.py --desde "Qwen3.5-2B"         # falla: excluido del roster activo
  ```

- [ ] Reemplazar el bloque **Bloqueo de infraestructura (2026-08-03)** por su estado real al
  2026-08-04: la imagen del grupo A ya se pudo construir y correr (la sonda de
  `Qwen2.5-1.5B-Instruct` corrió dentro de `slm-domotica-2026:granite-4-0-350m`), así que el daemon
  volvió a estado de escritura; lo que queda pendiente son las **3** reconstrucciones del grupo B y la
  **1** construcción nueva del grupo A (Tarea 7).
- [ ] Añadir, en la sección de continuación ante fallo, que el barrido **no se detiene** por un modelo:
  registra el fallo en `data/2026/fallos_barrido.json` con su error textual y sigue; un 401/403, en
  cambio, **sí** corta inmediatamente (AUTH-STOP) y no se reintenta.

### Tarea 6 — Ajustar los tests de coherencia README ↔ registro (TDD)

- [ ] Escribir/ajustar en `tests/test_docker_matriz.py`:

  ```python
  def test_tag_por_modelo_es_unico_y_usa_el_slug():
      tags = [tag_imagen(m) for m in MODELOS_2026]
      assert len(set(tags)) == 15
      assert tag_imagen(MODELOS_2026[0]) == f"slm-domotica-2026:{slug(MODELOS_2026[0].nombre)}"


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
  ```

- [ ] Correr y confirmar que **falla** antes de editar el README, y **verde** después:
  `pytest -q tests/test_docker_matriz.py`
- [ ] Correr la suite completa: `pytest -q`

### Tarea 7 — Imágenes Docker: 3 reconstrucciones del grupo B + 1 construcción nueva del grupo A

> Exención de TDD: son operaciones de build, no comportamiento nuevo. Su corrección se verifica
> leyendo la versión efectiva **dentro** de cada imagen.

> **Trasladada desde `TODO_16_roster-12-y-pins.md`, Tarea 6, con los conteos corregidos.** Los
> conteos viejos (4 imágenes del grupo B, 8 del grupo A que no se reconstruyen) quedaron numéricamente
> mal con el intercambio de roster, y la cola no podía quedarse en el nodo 16 sin crear un ciclo en el
> DAG (ver §3 del índice).

- [ ] Reconstruir las **3** imágenes del grupo B que están en el roster activo, una por una.
  `Qwen3.5-2B` **no** se construye: está excluido, y `build_all.py --modelo "Qwen3.5-2B"` debe fallar
  con `ValueError`:

  ```bash
  python docker/build_all.py --modelo "LFM2.5-230M"
  python docker/build_all.py --modelo "LFM2.5-350M"
  python docker/build_all.py --modelo "Qwen3.5-0.8B"
  python docker/build_all.py --modelo "Qwen3.5-2B"   # -> ValueError esperado, NO construye
  ```

- [ ] Construir la **1** imagen nueva del grupo A, la del modelo que entró (no existe todavía en este
  host):

  ```bash
  python docker/build_all.py --modelo "Qwen2.5-1.5B-Instruct"
  docker images --format '{{.Repository}}:{{.Tag}}' | grep '^slm-domotica-2026:qwen2-5-1-5b-instruct$'
  ```

- [ ] **No** reconstruir las **8** imágenes del grupo A que ya existen y cuyo pin
  (`BASELINE_TRANSFORMERS`) no cambió: `granite-4.0-350m`, `granite-4.0-h-350m`, `granite-4.0-1b`,
  `granite-4.0-h-1b`, `SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`, `LFM2.5-1.2B-Instruct`,
  `OLMo-2-0425-1B-Instruct`. Reconstruir sin necesidad viola el alcance de esta tarea. 8 ya
  construidas + 1 nueva = **9** imágenes del grupo A, que con las **3** del grupo B dan las **12** del
  roster activo.
- [ ] Confirmar la versión efectiva **dentro** de al menos una imagen de cada grupo, y de la imagen
  nueva:

  ```bash
  docker run --rm slm-domotica-2026:lfm2-5-230m python -c "import transformers; print(transformers.__version__)"
  # -> 5.x  (grupo B)
  docker run --rm slm-domotica-2026:granite-4-0-350m python -c "import transformers; print(transformers.__version__)"
  # -> 4.57.x  (grupo A, ya construida en el subtask 04)
  docker run --rm slm-domotica-2026:qwen2-5-1-5b-instruct python -c "import transformers; print(transformers.__version__)"
  # -> 4.57.x  (grupo A, imagen nueva)
  ```

- [ ] Confirmar que las 12 del roster activo están presentes:

  ```bash
  python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u > /tmp/plan.txt
  docker images --format '{{.Repository}}:{{.Tag}}' | grep '^slm-domotica-2026:' | sort -u > /tmp/hay.txt
  comm -23 /tmp/plan.txt /tmp/hay.txt   # -> vacio: no falta ninguna imagen del plan
  ```

- [ ] **Si Docker no está disponible o su almacenamiento vuelve a modo solo lectura**, dejar estas
  casillas sin marcar y anotar el error textual acá mismo. Las Tareas 1–6 y 8 no dependen de Docker.
  **No** intentar reparar Docker Desktop desde acá: la máquina la usa el usuario.
- [ ] **Si aparece un 401/403 en cualquier build o run** → **PARAR**, no reintentar, no ejecutar ningún
  comando de `login`/`auth`/`configure`, y reportar `AUTH-BLOCKER` con el error textual. Ninguno de los
  12 del roster activo es gated, así que un 403 acá indicaría que se coló un modelo excluido.

### Tarea 8 — Cierre

- [ ] `pytest -q` — suite completa verde.
- [ ] `python docker/build_all.py --dry-run | grep -c 'build slm-domotica-2026:'` → **12**
- [ ] `python docker/build_all.py --dry-run | grep -c 'transformers>=4.57.0,<5.0.0'` → **9**
- [ ] `python docker/build_all.py --dry-run | grep -c 'transformers>=5.0.0'` → **3**
- [ ] `python docker/run_sweep.py --dry-run | grep -c -- '--cpuset-cpus=0-1'` → **12**
- [ ] `python docker/run_sweep.py --dry-run | grep -c -- '--env-file'` → **0**
- [ ] `git grep -iE 'hf_[A-Za-z0-9]{20,}'` → vacío (exit 1).
- [ ] `git grep -n 'Qwen2.5-1.7B-Instruct'` → vacío (exit 1): el ID inexistente no quedó en ningún lado.
- [ ] F0 intacto:
  ```bash
  git diff --stat main -- data/resultados_experimento_detalle.csv \
    data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv \
    tests/test_metricas.py src/models.py src/prompt.py src/scoring.py src/schema.py \
    src/metrics.py src/run_evaluation.py src/generate_figures.py \
    src/validar_contra_resultados_originales.py
  ```
  → vacío.
- [ ] `git diff --stat -- data/2026/detalle/granite-4-0-350m.csv` → vacío (no se recomputó).
- [ ] `git add src/models_2026.py docker/run_sweep.py docker/README.md tests/test_models_2026.py tests/test_docker_matriz.py tests/test_credenciales_gated.py`
- [ ] Commitear:
  ```
  git commit -m "feat(2026): intercambio de roster, pinning de nucleos y continuacion ante fallo"
  ```

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Registro de 15, roster activo de 12 (6/6), 3 excluidos, 2 gated, 13 no-gated
python -c "
import sys; sys.path.insert(0,'src')
from models_2026 import MODELOS_2026, gated, roster_activo
r = roster_activo()
assert len(MODELOS_2026) == 15, len(MODELOS_2026)
assert len(r) == 12, len(r)
assert sum(1 for m in r if m.tier == 'sub-1B') == 6
assert sum(1 for m in r if m.tier == '1-2B') == 6
assert not any(m.gated for m in r)
assert len(gated()) == 2
assert len([m for m in MODELOS_2026 if not m.activo]) == 3
assert len([m for m in MODELOS_2026 if not m.gated]) == 13
print('conteos OK: registro 15, activo 12 (6/6), excluidos 3, gated 2, no-gated 13')
"

# 3. Grupos de version: 9 A / 3 B sobre el roster activo, 11 A / 4 B sobre el registro
python -c "
import sys; sys.path.insert(0,'src')
from models_2026 import BASELINE_TRANSFORMERS, MODELOS_2026, TRANSFORMERS_5X, roster_activo
r = roster_activo()
assert len([m for m in r if m.transformers_pin == BASELINE_TRANSFORMERS]) == 9
assert len([m for m in r if m.transformers_pin == TRANSFORMERS_5X]) == 3
assert len([m for m in MODELOS_2026 if m.transformers_pin == BASELINE_TRANSFORMERS]) == 11
assert len([m for m in MODELOS_2026 if m.transformers_pin == TRANSFORMERS_5X]) == 4
print('grupos OK: activo A=9 B=3 | registro A=11 B=4')
"

# 4. El intercambio, fila por fila
python -c "
import sys; sys.path.insert(0,'src')
from models_2026 import BASELINE_TRANSFORMERS, TRANSFORMERS_5X, por_nombre
nuevo = por_nombre('Qwen2.5-1.5B-Instruct')
assert nuevo.hf_repo_id == 'Qwen/Qwen2.5-1.5B-Instruct'
assert nuevo.params_b == 1.54 and nuevo.tier == '1-2B'
assert nuevo.transformers_pin == BASELINE_TRANSFORMERS
assert nuevo.activo and not nuevo.gated and nuevo.motivo_pin == ''
viejo = por_nombre('Qwen3.5-2B')
assert not viejo.activo and not viejo.gated
assert viejo.transformers_pin == TRANSFORMERS_5X and viejo.motivo_pin
assert 'no verificada' in viejo.motivo_exclusion.lower() or 'nunca se obtuvo' in viejo.motivo_exclusion.lower()
assert 'falla bajo 5.14.1' not in viejo.motivo_exclusion.lower()
print('intercambio OK, y la redaccion de Qwen3.5-2B no sobreafirma')
"

# 5. El ID inexistente no esta en ningun lado del arbol trackeado
git grep -n 'Qwen2.5-1.7B-Instruct' ; test $? -eq 1 && echo "ID inexistente ausente OK"

# 6. Los 3 anclas de continuidad con el paper publicado (RF19)
python -c "
import json, sys; sys.path.insert(0,'src')
from models_2026 import por_nombre, roster_activo
pub = json.load(open('data/resultados_experimento_resumen.json', encoding='utf-8'))
por_pub = {f['modelo']: f for f in pub}
anclas = sorted({m.nombre for m in roster_activo()} & set(por_pub))
assert anclas == ['Qwen2.5-1.5B-Instruct', 'SmolLM2-1.7B-Instruct', 'SmolLM2-360M-Instruct'], anclas
for n in anclas:
    assert por_nombre(n).params_b == por_pub[n]['params_b'], n
    print(f'{n:26s} estricta publicada={por_pub[n][\"exact_match_pct\"]:5.1f}%  latencia={por_pub[n][\"avg_latencia_s\"]:7.3f}s')
print('3 anclas de continuidad OK (eran 2 antes del Delta 03)')
"

# 7. cpuset en las 12 invocaciones, un solo valor, sin GPU, sin --env-file
python docker/run_sweep.py --dry-run | grep -c -- '--cpuset-cpus=0-1'   # -> 12
python docker/run_sweep.py --dry-run | grep -oE -- '--cpuset-cpus=[^ ]+' | sort -u | wc -l   # -> 1
python docker/run_sweep.py --dry-run | grep -cE 'gpus|--device|nvidia' ; test $? -eq 1 && echo "sin GPU OK"
python docker/run_sweep.py --dry-run | grep -c -- '--env-file'          # -> 0
python docker/run_sweep.py --dry-run | grep -c '^==='                   # -> 12

# 8. Plan de build: 12 imagenes, 9 con el pin del grupo A y 3 con el del grupo B
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u | wc -l  # -> 12
python docker/build_all.py --dry-run | grep -c 'transformers>=4.57.0,<5.0.0'   # -> 9
python docker/build_all.py --dry-run | grep -c 'transformers>=5.0.0'           # -> 3

# 9. Pedir cualquiera de los 3 excluidos falla con su motivo, sin construir ni correr nada
for m in "gemma-3-270m-it" "Llama-3.2-1B-Instruct" "Qwen3.5-2B"; do
  python docker/build_all.py --modelo "$m" --dry-run >/dev/null 2>&1 ; echo "$m exit=$?"   # -> exit=1
done
python docker/run_sweep.py --desde "Qwen3.5-2B" --dry-run ; echo "exit=$?"   # -> exit distinto de 0

# 10. Continuacion ante fallo: probada por test, no por corrida real
pytest -q tests/test_docker_matriz.py -k "barrido or cpuset"

# 11. Coherencia README <-> registro
pytest -q tests/test_docker_matriz.py -k readme

# 12. Sin secretos versionados y F0 intacto
git grep -iE 'hf_[A-Za-z0-9]{20,}' ; test $? -eq 1 && echo "sin token versionado OK"
git diff --stat main -- data/resultados_experimento_detalle.csv \
  data/resultados_experimento_resumen.json data/dataset_comandos_domotica.csv \
  tests/test_metricas.py src/models.py src/prompt.py src/scoring.py src/schema.py \
  src/metrics.py src/run_evaluation.py src/generate_figures.py \
  src/validar_contra_resultados_originales.py

# 13. granite-4-0-350m.csv no se toco
git diff --stat -- data/2026/detalle/granite-4-0-350m.csv   # -> vacio
```

## Acceptance criteria

- **Dado** `src/models_2026.py`, **entonces** `MODELOS_2026` tiene exactamente **15** elementos, con
  nombres, `hf_repo_id` y `slug` únicos, y las 14 filas que ya existían en `864c694` conservan su
  posición y su orden relativo (la fila nueva va al final, así que ningún artefacto ordenado por roster
  se reordena).
- **Dado** `por_nombre("Qwen2.5-1.5B-Instruct")`, **entonces** devuelve
  `hf_repo_id == "Qwen/Qwen2.5-1.5B-Instruct"`, `params_b == 1.54` (igual que la Tabla 1 publicada),
  `tier == "1-2B"`, `transformers_pin == BASELINE_TRANSFORMERS`, `gated is False`,
  `trust_remote_code is False`, `activo is True`, `motivo_exclusion == ""` y
  **`motivo_pin == ""`** — su pin es *heredado*, porque la sonda bajo 5.14.1 no se corrió.
- **Dado** el árbol trackeado completo, **entonces** `git grep -n 'Qwen2.5-1.7B-Instruct'` sale vacío:
  el ID que el usuario escribió y que **no existe** (HTTP 404) no quedó en ningún archivo, y el registro
  tampoco contiene `Qwen/Qwen2.5-0.5B-Instruct`.
- **Dado** `por_nombre("Qwen3.5-2B")`, **entonces** `activo is False`, `gated is False`,
  `motivo_exclusion` es no vacío y menciona que la confirmación bajo 5.14.1 **no se obtuvo** /
  compatibilidad **no verificada**, **sin** afirmar en ningún momento que el modelo falle bajo 5.14.1;
  y conserva `transformers_pin == TRANSFORMERS_5X` con su `motivo_pin` no vacío, porque su pin del
  grupo B sigue siendo **necesario**.
- **Dado** `MODELOS_2026`, **entonces** los **tres** modelos con `activo is False` son
  `gemma-3-270m-it`, `Llama-3.2-1B-Instruct` y `Qwen3.5-2B`; los dos primeros citan el `403` de
  `/resolve/`, el tercero cita `5.14.1`; y sigue valiendo `activo == False` ⟺ `motivo_exclusion != ""`.
- **Dado** `roster_activo()`, **entonces** devuelve **12** modelos en orden de registro, **6 por tier**
  y **cero** `gated`: el intercambio es a **conteo constante** porque salió y entró un modelo del tier
  `1-2B`.
- **Dado** los grupos de versión, **entonces** el roster activo se reparte **9** en el grupo A y **3**
  en el grupo B, y el registro completo **11** en A y **4** en B; y `gated()` sigue devolviendo
  exactamente **2** modelos (el intercambio no toca el conteo de gated).
- **Dado** el conjunto de pines necesarios, **entonces** sobre el **registro** son **5**
  (`{LFM2.5-230M, LFM2.5-350M, Qwen3.5-0.8B, Qwen3.5-2B, granite-4.0-350m}`) y sobre el **roster
  activo** son **4** (los mismos menos `Qwen3.5-2B`); ningún test afirma un conjunto de necesarios sin
  decir sobre cuál de los dos cuenta.
- **Dado** `[m for m in MODELOS_2026 if not m.gated]`, **entonces** tiene **13** elementos (los 12
  activos más `Qwen3.5-2B`), y `validar_credenciales` sobre esa lista devuelve `None` **sin** `.env`:
  ningún modelo excluido-pero-no-gated dispara el fail-fast de credenciales.
- **Dado** `comando_run(m, raiz)` para cualquier modelo, **entonces** el comando incluye
  `--memory=8g`, `--cpus=2` y `--cpuset-cpus=0-1`; el valor de `--cpuset-cpus` es **idéntico en las 12
  invocaciones del roster activo**; se puede sobreescribir con `comando_run(..., cpuset="2-3")` y con
  `python docker/run_sweep.py --cpuset "2-3"`; y ninguna invocación menciona `--gpus`, `--device` ni
  `nvidia`.
- **Dado** `ejecutar_barrido` con un modelo que devuelve código distinto de 0, **entonces** los
  modelos siguientes **igual se corren** (ninguno se saltea), el fallo queda registrado en
  `data/2026/fallos_barrido.json` con exactamente las claves `modelo`, `hf_repo_id`,
  `transformers_pin`, `codigo_salida`, `error_textual`, `momento_iso`, el error textual se imprime a
  stderr en el momento, y la función devuelve `1` **después** de haber intentado todos.
- **Dado** un barrido sin fallos, **entonces** `data/2026/fallos_barrido.json` existe y contiene `[]`,
  y la función devuelve `0`; **dado** `--dry-run`, **entonces** el archivo **no** se escribe.
- **Dado** un `401`/`403` en cualquier momento, **entonces** el proceso se detuvo sin reintentar y sin
  ejecutar ningún comando de `login`/`auth`/`configure`, y quedó reportado como `AUTH-BLOCKER`: la
  continuación ante fallo aplica a fallos técnicos del modelo, **nunca** a fallos de autenticación.
- **Dado** `docker/README.md`, **entonces** (a) la matriz del grupo A lista **9** modelos incluida la
  fila nueva con su línea `PROBE_OK` verbatim y la aclaración de que el grupo se sondeó y no se
  infirió de la familia; (b) el grupo B lista **3** activos más `Qwen3.5-2B` marcado como registrado y
  excluido, conservando su pin y su `motivo_pin`; (c) la tabla de exclusiones tiene **3** filas con sus
  **dos causas distintas**; (d) figura `--cpuset-cpus=0-1` con el razonamiento de cuota-vs-pinning y de
  por qué la ejecución es secuencial; (e) ningún ejemplo de uso propone `--desde "Qwen3.5-2B"` como
  operación normal.
- **Dado** las imágenes, **entonces** existen las **12** del roster activo —las **8** del grupo A que
  ya estaban (no se reconstruyeron), la **1** nueva del grupo A
  (`slm-domotica-2026:qwen2-5-1-5b-instruct`) y las **3** del grupo B reconstruidas—, `transformers`
  dentro de una imagen de cada grupo devuelve `4.57.*` y `5.*` respectivamente, y **no** existe ni se
  intentó construir la imagen de `Qwen3.5-2B`. Si Docker no estuvo disponible, las casillas de la
  Tarea 7 quedaron sin marcar con el error textual anotado, y el subtask 05 no puede arrancar.
- **Dado** el repositorio al final, **entonces** `pytest -q` pasa, ningún archivo de F0 cambió,
  `data/2026/detalle/granite-4-0-350m.csv` sigue byte-idéntico,
  `git grep -iE 'hf_[A-Za-z0-9]{20,}'` sale vacío, y no se tocaron `src/run_sweep_2026.py`,
  `src/prompt_2026.py`, `src/judge_2026.py` ni `docker/build_all.py`.
