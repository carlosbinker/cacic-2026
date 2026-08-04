#!/usr/bin/env python3
"""Corre la corrida de CONTROL: los tres anclajes de continuidad, de a uno,
con el prompt ORIGINAL del paper en vez del prompt 2026 endurecido.

Mirroring de `docker/run_sweep.py` a propósito: mismo envolvente de recursos
(`--memory=8g --cpus=2 --cpuset-cpus=0-1`), misma decodificación de stderr en
UTF-8 (`encoding="utf-8", errors="replace"` -- el mismo fix que evitó que un
byte no-cp1252 tirara abajo el barrido principal), y la misma continuación
ante fallo (F12.1): un anclaje que falla se registra con su error textual en
`data/2026/control_prompt_original/fallos_control.json` (siempre existe
salvo `--dry-run`) y la corrida sigue con el siguiente.

Los tres anclajes son grupo A (`transformers==4.57.6`): sus imágenes ya
existen del barrido principal (`slm-domotica-2026:{smollm2-360m-instruct,
smollm2-1-7b-instruct,qwen2-5-1-5b-instruct}`) y **no se reconstruyen**. Este
script nunca invoca `docker build`.

Uso:
    python docker/run_control.py [--force] [--dry-run] [--cpuset PAR]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(AQUI))

from build_all import tag_imagen  # noqa: E402
from models_2026 import ModeloEvaluado2026, por_nombre  # noqa: E402
from run_control_prompt_original import ANCLAS_CONTROL, ruta_control  # noqa: E402

CPUSET_POR_DEFECTO = "0-1"
DIR_CONTROL = RAIZ / "data" / "2026" / "control_prompt_original"
FALLOS_PATH = DIR_CONTROL / "fallos_control.json"


def modelos_control() -> list[ModeloEvaluado2026]:
    """Los tres anclajes de continuidad, en el orden fijo de `ANCLAS_CONTROL`."""
    return [por_nombre(nombre) for nombre in ANCLAS_CONTROL]


def comando_run_control(modelo: ModeloEvaluado2026, raiz: Path, force: bool = False,
                        cpuset: str = CPUSET_POR_DEFECTO) -> list[str]:
    """El `docker run` de un anclaje contra el entrypoint de control.

    Mismo envolvente de recursos que `docker/run_sweep.py` (RNF1/RNF6).
    Ninguno de los tres anclajes es gated, así que no hay `--env-file`.
    Reusa la imagen ya construida del barrido principal: cero rebuild.
    """
    cmd = [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=2", f"--cpuset-cpus={cpuset}",
        "-v", f"{raiz / 'data'}:/app/data",
        "-v", f"{raiz / '.hf_cache'}:/app/.hf_cache",
        tag_imagen(modelo),
        "python",
    ]
    if force:
        cmd += ["src/run_control_prompt_original.py", "--force", "--modelo", modelo.nombre]
    else:
        cmd += ["src/run_control_prompt_original.py", "--modelo", modelo.nombre]
    return cmd


def _csv_ya_completo(ruta: Path) -> bool:
    """Chequeo host-side liviano: evita levantar un contenedor si el CSV de
    ese anclaje ya está completo y bien formado (mismo criterio que
    `run_sweep_2026.debe_saltear`, sin necesitar `--force` a nivel host)."""
    if not ruta.exists():
        return False
    import pandas as pd

    from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO
    try:
        df = pd.read_csv(ruta)
    except Exception:
        return False
    if list(df.columns) != COLUMNAS_DETALLE or len(df) != N_COMANDOS_ESPERADO:
        return False
    try:
        indices = sorted(int(i) for i in df["idx"])
    except (TypeError, ValueError):
        return False
    return indices == list(range(N_COMANDOS_ESPERADO))


def _correr_modelo(modelo: ModeloEvaluado2026, posicion: str, force: bool,
                   dry_run: bool, cpuset: str) -> tuple[int, str]:
    """Corre el contenedor de un anclaje; devuelve (código de salida, stderr)."""
    cmd = comando_run_control(modelo, RAIZ, force, cpuset)
    print(f"\n=== [{posicion}] control::{modelo.nombre} ===")
    print(" ".join(cmd))
    if dry_run:
        return 0, ""
    # Mismo fix que docker/run_sweep.py: sin `encoding` explícito, `text=True`
    # decodifica con la codificación preferida del locale del host (cp1252 en
    # Windows), que revienta con bytes UTF-8 fuera de su rango.
    completado = subprocess.run(
        cmd, stderr=subprocess.PIPE, encoding="utf-8", errors="replace"
    )
    if completado.stderr:
        print(completado.stderr, file=sys.stderr)
    return completado.returncode, completado.stderr or ""


def _registrar_fallo(modelo: ModeloEvaluado2026, codigo: int, error: str) -> dict:
    """La fila que documenta un fallo irrecuperable de un anclaje."""
    return {
        "modelo": modelo.nombre,
        "hf_repo_id": modelo.hf_repo_id,
        "transformers_pin": modelo.transformers_pin,
        "codigo_salida": codigo,
        "error_textual": error,
        "momento_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ejecutar_control(modelos: list[ModeloEvaluado2026], force: bool, dry_run: bool,
                     cpuset: str = CPUSET_POR_DEFECTO) -> int:
    """De a un contenedor por vez, en el orden de `ANCLAS_CONTROL`; no corta
    al primer fallo (mismo contrato F12.1 que `docker/run_sweep.py`)."""
    fallos: list[dict] = []
    try:
        for i, modelo in enumerate(modelos, 1):
            ruta = ruta_control(modelo.nombre)
            if not force and not dry_run and _csv_ya_completo(ruta):
                print(f"[{i}/{len(modelos)}] {modelo.nombre}: ya completo en {ruta}, se saltea "
                      f"(usá --force)")
                continue
            try:
                codigo, error = _correr_modelo(modelo, f"{i}/{len(modelos)}", force,
                                               dry_run, cpuset)
            except Exception as exc:
                codigo, error = 1, f"{type(exc).__name__}: {exc}"
            if codigo != 0:
                fallos.append(_registrar_fallo(modelo, codigo, error))
                print(f"FALLO {modelo.nombre} (código {codigo}). Se registra y se "
                      f"CONTINÚA con el siguiente anclaje.\n{error}", file=sys.stderr)
    finally:
        if not dry_run:
            FALLOS_PATH.parent.mkdir(parents=True, exist_ok=True)
            FALLOS_PATH.write_text(
                json.dumps(fallos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    if fallos:
        print(f"\nCorrida de control terminada con {len(fallos)} fallo(s): "
              f"{', '.join(f['modelo'] for f in fallos)}. Detalle en {FALLOS_PATH}.",
              file=sys.stderr)
        return 1
    print("\nCorrida de control completa, sin fallos.")
    return 0


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrida de control: prompt original del paper sobre los 3 anclajes"
    )
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV del anclaje ya esté completo")
    parser.add_argument("--dry-run", action="store_true",
                        help="imprimir los comandos sin ejecutarlos")
    parser.add_argument("--cpuset", default=CPUSET_POR_DEFECTO,
                        help="par de núcleos al que se fija cada contenedor "
                             "(--cpuset-cpus); el mismo que el barrido principal")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    return ejecutar_control(modelos_control(), args.force, args.dry_run, args.cpuset)


if __name__ == "__main__":
    sys.exit(main())
