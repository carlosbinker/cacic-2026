#!/usr/bin/env python3
"""Construye una imagen por modelo del roster ACTIVO, con su pin de transformers.

Una imagen por modelo (RF4): los árboles de dependencias quedan aislados,
pero la caché de pesos es un único volumen compartido en runtime.

Este script nunca recibe ni declara credenciales: el único `--build-arg` es
`TRANSFORMERS_PIN`, que es una spec de versión, no un secreto (F11).

Selecciona por defecto `roster_activo()` (12), no el registro completo (14):
los dos modelos gated quedan fuera porque el acceso de descarga no fue
otorgado (ver `motivo_exclusion` en `src/models_2026.py`). Pedir uno de esos
dos explícitamente con `--modelo` falla con `ValueError`.

Uso:
    python docker/build_all.py [--modelo NOMBRE] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from models_2026 import ModeloEvaluado2026, por_nombre, roster_activo, slug  # noqa: E402

DOCKERFILE = RAIZ / "docker" / "Dockerfile.modelo"


def tag_imagen(modelo: ModeloEvaluado2026) -> str:
    """Nombre de imagen congelado en F11: `slm-domotica-2026:<slug>`."""
    return f"slm-domotica-2026:{slug(modelo.nombre)}"


def comando_build(modelo: ModeloEvaluado2026, raiz: Path) -> list[str]:
    """El `docker build` de ese modelo, con su pin inyectado por build-arg."""
    return [
        "docker", "build",
        "-f", str(raiz / "docker" / "Dockerfile.modelo"),
        "--build-arg", f"TRANSFORMERS_PIN={modelo.transformers_pin}",
        "-t", tag_imagen(modelo),
        str(raiz),
    ]


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


def construir(modelo: ModeloEvaluado2026, dry_run: bool) -> int:
    """Construye la imagen de un modelo y devuelve su código de salida."""
    cmd = comando_build(modelo, RAIZ)
    print(f"\n=== build {tag_imagen(modelo)} | {modelo.transformers_pin} ===")
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def construir_todas(modelos: list[ModeloEvaluado2026], dry_run: bool) -> int:
    """Construye en orden de roster y corta en el primer fallo."""
    for modelo in modelos:
        codigo = construir(modelo, dry_run)
        if codigo != 0:
            print(f"FALLO el build de {modelo.nombre}", file=sys.stderr)
            return codigo
    return 0


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye una imagen Docker por modelo del roster ACTIVO (12) del 2026"
    )
    parser.add_argument("--modelo", default=None,
                        help="construir solo esta imagen (ver src/models_2026.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="imprimir los comandos sin ejecutarlos")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    return construir_todas(seleccionar_modelos(args.modelo), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
