---
id: 04
title: Imágenes Docker por modelo y matriz de versiones
depends_on: [01, 03]
files:
  - docker/Dockerfile.modelo
  - docker/requirements-base.txt
  - docker/build_all.py
  - docker/run_sweep.py
  - docker/README.md
  - src/models_2026.py
  - tests/test_docker_matriz.py
  - .dockerignore
---

## Spec

Implementar **F11**: una imagen Docker por modelo, parametrizada por `ARG TRANSFORMERS_PIN`, compartiendo un único volumen de caché HF, ejecutadas de a una y en orden de roster con `--memory=8g --cpus=2` (**RF4**, **RNF1**).

Cubre además **RF5**, el requisito explícito del usuario: toda divergencia de versión respecto de `BASELINE_TRANSFORMERS` —y todo uso de `trust_remote_code`— queda documentada con su motivo en `docker/README.md`, en el sitio del pin (`src/models_2026.py`), y más adelante en el paper (subtasks 11, 14 y 15 la consumen desde el mismo registro).

Este subtask es el **único autorizado a modificar `src/models_2026.py`**, y solo los campos `transformers_pin`, `trust_remote_code` y `motivo_pin`, y solo con evidencia empírica de fallo. El `Dockerfile` legacy de la raíz queda intacto.

Cubre también la mitad "Docker" de **F11**: dos de los 14 modelos (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) son `gated`. El token de Hugging Face vive solo en `.env` en la raíz del repo (ignorado por git) y se inyecta **únicamente en tiempo de ejecución**, vía `docker run --env-file .env`, y **solo** a esas dos imágenes; las otras 12 corren sin credenciales. Prohibido en cualquier `Dockerfile` o script bajo `docker/`: `ARG HF_TOKEN`, `ENV HF_TOKEN`, `COPY .env`, pasar el token por `--build-arg`, o invocar `huggingface-cli login` / `huggingface_hub.login()`. `run_sweep.py` valida al arrancar que exista `.env` con `HF_TOKEN` no vacío si la lista a correr incluye algún `gated`, y aborta antes de construir o correr nada si falta.

## Implementation plan

### Tarea 1 — Imagen parametrizada y `.dockerignore`

> Exención de TDD: edición de configuración pura.

- [ ] Crear `docker/requirements-base.txt` (todo menos `transformers`, que se inyecta por `ARG`):

```
torch>=2.4.0
pandas>=2.2.0
numpy>=1.26.0
accelerate>=1.0.0
sentencepiece>=0.2.0
```

- [ ] Crear `docker/Dockerfile.modelo`:

```dockerfile
# Imagen de evaluación de UN modelo del roster 2026.
#
# Se construye una por modelo para que los árboles de dependencias queden
# aislados: varios modelos del roster (Granite 4.0 hibridos, LFM2.5, Qwen3.5)
# son releases recientes y pueden exigir versiones distintas de transformers.
# El pin efectivo de cada uno vive en src/models_2026.py y se documenta en
# docker/README.md (ver RF5 del TODO).
#
# La caché de pesos SÍ se comparte entre imágenes (un solo volumen .hf_cache):
# aísla dependencias, no descargas.

FROM python:3.11-slim

ARG TRANSFORMERS_PIN="transformers>=4.57.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt \
    && pip install --no-cache-dir "${TRANSFORMERS_PIN}"

COPY src/ /app/src/
COPY data/dataset_comandos_domotica.csv /app/data/dataset_comandos_domotica.csv

ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/run_sweep_2026.py", "--listar"]
```

- [ ] Crear `.dockerignore` para que el contexto de build no arrastre pesos ni PDFs:

```
.git
.hf_cache
.claude-scratch
data/2026
figures
paper
LaTeX2e*
*.docx
*.pdf
__pycache__
.pytest_cache
```

### Tarea 2 — Constructor y corredor (TDD sobre la generación de comandos)

- [ ] Escribir el test que falla, `tests/test_docker_matriz.py`:

```python
"""Tests de la orquestación Docker y del contrato de documentación de pines."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))

from build_all import comando_build, tag_imagen  # noqa: E402
from models_2026 import BASELINE_TRANSFORMERS, MODELOS_2026, slug  # noqa: E402
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
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_docker_matriz.py`
- [ ] Implementar `docker/build_all.py`:

```python
#!/usr/bin/env python3
"""Construye una imagen por modelo del roster, con su pin de transformers.

Uso:
    python docker/build_all.py [--modelo NOMBRE] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from models_2026 import MODELOS_2026, ModeloEvaluado2026, por_nombre, slug  # noqa: E402

DOCKERFILE = RAIZ / "docker" / "Dockerfile.modelo"


def tag_imagen(modelo: ModeloEvaluado2026) -> str:
    return f"slm-domotica-2026:{slug(modelo.nombre)}"


def comando_build(modelo: ModeloEvaluado2026, raiz: Path) -> list[str]:
    return [
        "docker", "build",
        "-f", str(raiz / "docker" / "Dockerfile.modelo"),
        "--build-arg", f"TRANSFORMERS_PIN={modelo.transformers_pin}",
        "-t", tag_imagen(modelo),
        str(raiz),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modelos = [por_nombre(args.modelo)] if args.modelo else MODELOS_2026
    for modelo in modelos:
        cmd = comando_build(modelo, RAIZ)
        print(f"\n=== build {tag_imagen(modelo)} | {modelo.transformers_pin} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        resultado = subprocess.run(cmd)
        if resultado.returncode != 0:
            print(f"FALLO el build de {modelo.nombre}", file=sys.stderr)
            return resultado.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Implementar `docker/run_sweep.py`:

```python
#!/usr/bin/env python3
"""Corre el barrido completo: las 14 imágenes, de a una, en orden de roster.

Secuencial a propósito (RF4): dos modelos en paralelo se pelearían por los
2 núcleos y contaminarían la tabla de latencia, que es el resultado central
del paper. El harness es reanudable, así que reejecutar este script después
de una interrupción retoma donde quedó.

Credenciales (F11): dos de los 14 modelos son gated. Reciben el token de
Hugging Face SOLO en tiempo de ejecución, vía `docker run --env-file .env`;
las otras 12 imágenes corren sin credenciales. Si la lista a correr incluye
algún gated y falta `.env` o `HF_TOKEN`, este script aborta antes de correr
nada; nunca imprime el valor del token.

Uso:
    python docker/run_sweep.py [--desde NOMBRE] [--force] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from build_all import tag_imagen  # noqa: E402
from models_2026 import MODELOS_2026, ModeloEvaluado2026  # noqa: E402


def comando_run(modelo: ModeloEvaluado2026, raiz: Path, force: bool = False) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=2",
    ]
    if modelo.gated:
        # F11: exclusivamente estos dos modelos reciben el token, y solo así.
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


def _hf_token_de_env(ruta_env: Path) -> str:
    """Lee HF_TOKEN de .env sin nunca imprimirlo. Devuelve "" si no está."""
    for linea in ruta_env.read_text(encoding="utf-8").splitlines():
        if linea.strip().startswith("HF_TOKEN="):
            return linea.split("=", 1)[1].strip()
    return ""


def validar_credenciales(modelos: list[ModeloEvaluado2026], raiz: Path) -> None:
    """F11: si la lista a correr incluye algún gated, aborta ANTES de construir
    o correr nada si falta .env o HF_TOKEN. El mensaje nunca imprime el token."""
    if not any(m.gated for m in modelos):
        return
    ruta_env = raiz / ".env"
    if not ruta_env.exists():
        raise ValueError(
            "La lista a correr incluye modelos gated pero no existe .env en la "
            "raíz del repo. Creá .env con HF_TOKEN=<tu token> (ver docker/README.md)."
        )
    if not _hf_token_de_env(ruta_env):
        raise ValueError(
            "La lista a correr incluye modelos gated pero HF_TOKEN está ausente "
            "o vacío en .env. Completalo (ver docker/README.md)."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", default=None, help="retomar desde este nombre de modelo")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modelos = list(MODELOS_2026)
    if args.desde:
        nombres = [m.nombre for m in modelos]
        if args.desde not in nombres:
            raise ValueError(f"{args.desde!r} no está en el roster: {nombres}")
        modelos = modelos[nombres.index(args.desde):]

    validar_credenciales(modelos, RAIZ)

    for i, modelo in enumerate(modelos, 1):
        cmd = comando_run(modelo, RAIZ, args.force)
        print(f"\n=== [{i}/{len(modelos)}] {modelo.nombre} ({modelo.tier}) ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        resultado = subprocess.run(cmd)
        if resultado.returncode != 0:
            print(f"FALLO {modelo.nombre} (código {resultado.returncode}). "
                  f"Corregí y retomá con --desde \"{modelo.nombre}\".", file=sys.stderr)
            return resultado.returncode
    print("\nBarrido completo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> Nota: `comando_run` construye `["...", "--modelo", nombre]` al final en ambos casos, de modo que `cmd[-3:] == ["src/run_sweep_2026.py", "--modelo", nombre]` solo se cumple sin `--force`; el test lo ejerce sin `--force`.

### Tarea 3 — Descubrir los pines necesarios y documentarlos (RF5)

> Exención de TDD: es un procedimiento empírico de descubrimiento. Su resultado queda blindado por los tests de la Tarea 2 (`test_todo_pin_divergente_esta_justificado_en_el_readme`).

Para **cada** modelo del roster, en orden:

- [ ] Construir con el pin actual: `python docker/build_all.py --modelo "<nombre>"`
- [ ] Probar carga y una sola generación dentro del contenedor. Para los 12 modelos no gated, sin credenciales:

```bash
docker run --rm --memory=8g --cpus=2 \
  -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  slm-domotica-2026:<slug> \
  python -c "
import sys; sys.path.insert(0,'src')
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from models_2026 import por_nombre
from prompt_2026 import construir_entrada
m = por_nombre('<nombre>')
tok = AutoTokenizer.from_pretrained(m.hf_repo_id, trust_remote_code=m.trust_remote_code)
red = AutoModelForCausalLM.from_pretrained(m.hf_repo_id, device_map='cpu',
                                           trust_remote_code=m.trust_remote_code)
e, modo = construir_entrada(tok, 'Prendé la luz del living.')
print('OK', transformers.__version__, modo, red.generate(**e, max_new_tokens=8).shape)
"
```

  Para los dos modelos gated (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`), la única diferencia es `--env-file .env` en el `docker run` y `token=os.environ["HF_TOKEN"]` en ambos `from_pretrained` (F11; nunca `login()`):

```bash
docker run --rm --memory=8g --cpus=2 \
  --env-file "$(pwd)/.env" \
  -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  slm-domotica-2026:<slug> \
  python -c "
import os, sys; sys.path.insert(0,'src')
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from models_2026 import por_nombre
from prompt_2026 import construir_entrada
m = por_nombre('<nombre>')
tok = AutoTokenizer.from_pretrained(m.hf_repo_id, trust_remote_code=m.trust_remote_code,
                                    token=os.environ['HF_TOKEN'])
red = AutoModelForCausalLM.from_pretrained(m.hf_repo_id, device_map='cpu',
                                           trust_remote_code=m.trust_remote_code,
                                           token=os.environ['HF_TOKEN'])
e, modo = construir_entrada(tok, 'Prendé la luz del living.')
print('OK', transformers.__version__, modo, red.generate(**e, max_new_tokens=8).shape)
"
```

- [ ] **Si funciona:** no tocar nada. El modelo hereda el baseline y `motivo_pin` queda `""`.
- [ ] **Si falla**, registrar el error textual y actuar según el caso:
  - `ValueError: The checkpoint you are trying to load has model type 'X' but Transformers does not recognize this architecture` → subir a la primera versión que sí lo soporta.
  - Se pide explícitamente `trust_remote_code` → poner `trust_remote_code=True` para ese modelo.
  - Incompatibilidad de API entre `transformers` y `torch` → fijar la versión que resuelve.
- [ ] Editar **solo** los campos `transformers_pin`, `trust_remote_code` y `motivo_pin` de ese modelo en `src/models_2026.py`, con un comentario en el sitio del pin. Formato del motivo: `"<versión/flag> requerido por <modelo>: <error textual abreviado>"`. Ejemplo del formato esperado:

```python
    ModeloEvaluado2026(
        nombre="granite-4.0-h-1b",
        hf_repo_id="ibm-granite/granite-4.0-h-1b",
        params_b=1.5,
        tier="1-2B",
        # Pin divergente (RF5): la arquitectura híbrida Mamba-2 no está en el
        # baseline; con transformers>=4.57.0 falla al construir el modelo.
        transformers_pin="transformers==4.58.2",
        trust_remote_code=False,
        motivo_pin=(
            "transformers==4.58.2 requerido por granite-4.0-h-1b: con el baseline "
            "falla con \"does not recognize this architecture (granitemoehybrid)\""
        ),
    ),
```

- [ ] Si el subtask 04 introduce algún pin divergente, **eliminar** `test_estado_inicial_sin_divergencias` de `tests/test_models_2026.py` (era una foto del estado inicial, ver nota del subtask 01). `test_invariante_motivo_pin` **no** se toca nunca.
- [ ] Escribir `docker/README.md` con esta estructura exacta:

```markdown
# Aislamiento por modelo y matriz de versiones

Cada modelo del roster 2026 corre en su propia imagen, construida desde
`Dockerfile.modelo` con su propio `--build-arg TRANSFORMERS_PIN`. Las imágenes
comparten un único volumen de caché de pesos (`.hf_cache`): se aíslan los
árboles de dependencias, no las descargas.

Baseline del proyecto: `transformers>=4.57.0`.

## Matriz de versiones

| Modelo | `transformers_pin` | `trust_remote_code` | ¿Divergente? | Motivo | ¿Gated? |
|---|---|---|---|---|---|
| ... una fila por cada uno de los 14 modelos ... |

Los modelos marcados como no divergentes heredan el baseline; su pin no es una
decisión, es la ausencia de una. Solo las filas marcadas como divergentes
representan una restricción real descubierta empíricamente, y son las que se
reportan en la Tabla 5 y en la Sección 6 (Amenazas a la validez) del paper:
correr distintos modelos con distintas versiones de la librería de inferencia
es un confusor para la comparación de latencia.

> Nota informativa: Gemma 3 requiere `transformers >= 4.50.0`; cubierto por el
> baseline `transformers>=4.57.0`. No es un pin divergente — `gemma-3-270m-it`
> hereda el baseline igual que los demás.

## Credenciales de los modelos gated

Dos modelos del roster son *gated*: `google/gemma-3-270m-it` y
`meta-llama/Llama-3.2-1B-Instruct` (columna "¿Gated?" de la tabla). El token
de Hugging Face vive **solo** en `.env` en la raíz del repo (ignorado por
git) y se inyecta **únicamente en tiempo de ejecución**, vía
`docker run --env-file .env`, y **solo** a esas dos imágenes; las otras 12
corren sin credenciales.

Prohibido: declarar `ARG HF_TOKEN` o `ENV HF_TOKEN` en `Dockerfile.modelo`,
pasar el token como `--build-arg`, hacer `COPY .env`, escribirlo en una capa
de imagen, o invocar `huggingface-cli login` / `huggingface_hub.login()`. La
única forma de consumirlo es leer `os.environ["HF_TOKEN"]` en runtime (ver
`src/run_sweep_2026.py`, subtask 03).

Si la lista a correr incluye algún modelo gated y `.env` no existe o
`HF_TOKEN` está ausente o vacío, `docker/run_sweep.py` aborta con un
`ValueError` en español antes de construir o correr nada; el mensaje nunca
imprime el valor del token.

## Uso

    python docker/build_all.py            # construye las 14 imágenes
    python docker/run_sweep.py            # corre el barrido completo, de a una
    python docker/run_sweep.py --desde "Qwen3.5-2B"   # retoma tras una interrupción
```

- [ ] Completar la tabla con las 14 filas reales (una por modelo, con su pin, flag, "sí"/"no", motivo o "—" y "sí"/"no" en ¿Gated?; los dos gated son `gemma-3-270m-it` y `Llama-3.2-1B-Instruct`).
- [ ] Correr y confirmar **verde**: `pytest -q tests/test_docker_matriz.py`

### Tarea 4 — commit

- [ ] `pytest -q`
- [ ] `git add docker/ .dockerignore src/models_2026.py tests/test_docker_matriz.py tests/test_models_2026.py`
- [ ] `git commit -m "feat(2026): una imagen Docker por modelo y matriz de versiones documentada"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Los comandos generados son los congelados en F11 (sin ejecutar docker)
python docker/build_all.py --dry-run | head -40
python docker/run_sweep.py --dry-run | grep -c "memory=8g"      # -> 14
python docker/run_sweep.py --dry-run | grep -c -- "--env-file"  # -> 2 (solo los gated)

# 3. Las 14 imágenes existen
docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^slm-domotica-2026:'   # -> 14

# 4. RF5: coherencia entre el registro, el README y los motivos
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from models_2026 import MODELOS_2026, BASELINE_TRANSFORMERS
readme = Path('docker/README.md').read_text(encoding='utf-8')
div = [m for m in MODELOS_2026 if m.transformers_pin != BASELINE_TRANSFORMERS]
for m in MODELOS_2026:
    assert m.nombre in readme, m.nombre
    assert bool(m.motivo_pin) == (m.transformers_pin != BASELINE_TRANSFORMERS), m.nombre
for m in div:
    assert m.transformers_pin in readme and m.motivo_pin[:30] in readme, m.nombre
print(f'{len(div)} pines divergentes, todos documentados')
"

# 5a. Prohibido: HF_TOKEN en capa de imagen, COPY .env, login interactivo
grep -rnE "ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env" docker/ \
  && echo "FALLA: credenciales en capa" || echo "sin credenciales en capa OK"
grep -rniE "huggingface-cli login|huggingface_hub\.login\(" docker/ \
  && echo "FALLA: login interactivo" || echo "sin login interactivo OK"

# 5b. Prohibido en todo el árbol trackeado: el VALOR del token
git grep -iE 'hf_[A-Za-z0-9]{20,}' \
  && echo "FALLA: token versionado" || echo "sin token versionado OK"

# 5c. Permitido y esperado: --env-file en el run de los dos modelos gated
python docker/run_sweep.py --dry-run | grep -c -- "--env-file"   # -> 2

# 6. El Dockerfile legacy de la raíz no fue tocado
git diff --exit-code main -- Dockerfile requirements.txt && echo "legacy intacto"

# 7. Solo cambiaron los tres campos permitidos de models_2026.py
git diff main -- src/models_2026.py | grep '^[-+]' | grep -vE '^[-+]{3}' \
  | grep -vE 'transformers_pin|trust_remote_code|motivo_pin|^\+\s*#' \
  && echo "FALLA: se tocaron otros campos" || echo "solo campos permitidos OK"
```

## Acceptance criteria

- **Dado** `docker/build_all.py --dry-run`, **entonces** emite 14 comandos `docker build`, cada uno con `-f docker/Dockerfile.modelo`, `--build-arg TRANSFORMERS_PIN=<pin del modelo>` y `-t slm-domotica-2026:<slug>` con 14 tags distintos.
- **Dado** `docker/run_sweep.py --dry-run`, **entonces** emite 14 comandos `docker run` **secuenciales**, todos con `--rm --memory=8g --cpus=2`, montando `data` en `/app/data` y una **única** caché compartida en `/app/.hf_cache`, e invocando `src/run_sweep_2026.py --modelo "<nombre>"`. Exactamente **dos** de esos comandos —los de `gemma-3-270m-it` y `Llama-3.2-1B-Instruct`— incluyen además `--env-file <repo>/.env`; los otros 12 no.
- **Dado** un fallo en el modelo N, **cuando** se relanza con `--desde "<nombre del modelo N>"`, **entonces** retoma en ese modelo y los anteriores no se rehacen (los saltea el harness del subtask 03).
- **Dado** cada uno de los 14 modelos, **cuando** se lo carga y genera 8 tokens dentro de su imagen (con `--env-file .env` para los dos gated), **entonces** no lanza excepción; si la lanzó, su `transformers_pin` y/o `trust_remote_code` fueron ajustados y `motivo_pin` cita el error textual.
- **Dado** el registro y `docker/README.md`, **entonces** los 14 modelos aparecen en la tabla (incluida la columna ¿Gated?), y para **todo** modelo con `transformers_pin != BASELINE_TRANSFORMERS` se cumple que `motivo_pin` es no vacío, y tanto el pin como el motivo figuran en el README. Lo mismo para todo modelo con `trust_remote_code=True`. `gemma-3-270m-it` **no** diverge: hereda el baseline, y la nota informativa sobre su mínimo de `transformers` no cuenta como divergencia.
- **Dado** cualquier archivo bajo `docker/`, **entonces** no declara `ARG HF_TOKEN` ni `ENV HF_TOKEN`, no hace `COPY .env`, no pasa el token por `--build-arg`, y no invoca `huggingface-cli login` ni `huggingface_hub.login()`; el **valor** del token tampoco aparece en ningún archivo versionado del repo (`git grep -iE 'hf_[A-Za-z0-9]{20,}'` vacío). Los dos modelos gated sí incluyen `--env-file .env` en su invocación de `docker run`; las otras 12 no.
- **Dado** una lista de modelos a correr que incluye algún gated, **cuando** `.env` no existe o `HF_TOKEN` está ausente/vacío en él, **entonces** `docker/run_sweep.py` aborta con `ValueError` en español antes de construir o correr nada, sin imprimir el valor del token.
- **Dado** `git diff main -- src/models_2026.py`, **entonces** las únicas líneas cambiadas corresponden a `transformers_pin`, `trust_remote_code`, `motivo_pin` o comentarios; el roster, los nombres, repos, params y tiers quedan iguales.
- **Dado** `Dockerfile` y `requirements.txt` de la raíz, **entonces** siguen byte-idénticos a `main`.
