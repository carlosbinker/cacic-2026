#!/usr/bin/env python3
"""Corre el barrido: las 12 imágenes del roster activo, de a una, en orden de registro.

Secuencial a propósito (RF4): dos modelos en paralelo se pelearían por los
2 núcleos y contaminarían la tabla de latencia, que es el resultado central
del paper. El harness es reanudable, así que reejecutar este script después
de una interrupción retoma donde quedó: los modelos con su CSV de detalle ya
completo los saltea `src/run_sweep_2026.py` (RF3), y `--desde` permite además
arrancar directamente en el modelo que falló.

Selecciona por defecto `roster_activo()` (12), no el registro completo (14):
los dos modelos gated quedan fuera porque el acceso de descarga no fue
otorgado. Retomar explícitamente desde uno de esos dos con `--desde` falla
con `ValueError`.

Credenciales (F11): el roster activo no tiene ningún modelo gated, así que
corre sin credenciales. La invocación con `--env-file .env` se conserva tal
cual para los dos modelos gated del registro, por si alguno se reactiva; si
la lista a correr incluyera algún gated y faltara `.env` o `HF_TOKEN`, este
script aborta antes de correr nada; nunca imprime el valor del token.

Uso:
    python docker/run_sweep.py [--desde NOMBRE] [--force] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
# `docker/` no es un paquete: se agregan ambas carpetas para que el módulo se
# pueda importar tanto como script (`python docker/run_sweep.py`) como por
# ruta (tests/test_credenciales_gated.py lo carga con importlib).
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(AQUI))

from build_all import tag_imagen  # noqa: E402
from models_2026 import ModeloEvaluado2026, por_nombre, roster_activo  # noqa: E402

CLAVE_TOKEN = "HF_TOKEN"


# --------------------------------------------------------------------------
# Construcción del comando (congelada en F11)
# --------------------------------------------------------------------------

def comando_run(modelo: ModeloEvaluado2026, raiz: Path,
                force: bool = False) -> list[str]:
    """El `docker run` congelado de ese modelo.

    Los límites `--memory=8g --cpus=2` no son negociables (RNF1): reproducen
    el hardware acotado del paper original y son lo que hace comparable la
    tabla de latencia.
    """
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


# --------------------------------------------------------------------------
# Credenciales: fallo temprano (F11 / RF17)
# --------------------------------------------------------------------------

def _hf_token_de_env(ruta_env: Path) -> str:
    """Lee HF_TOKEN de .env sin nunca imprimirlo. Devuelve "" si no está."""
    for linea in ruta_env.read_text(encoding="utf-8").splitlines():
        if linea.strip().startswith(f"{CLAVE_TOKEN}="):
            return linea.split("=", 1)[1].strip()
    return ""


def validar_credenciales(modelos: list[ModeloEvaluado2026], raiz: Path) -> None:
    """F11: si la lista a correr incluye algún gated, aborta ANTES de construir
    o correr nada si falta .env o HF_TOKEN. El mensaje nombra los modelos y el
    archivo faltante, pero nunca imprime el valor del token."""
    gated = [m.nombre for m in modelos if m.gated]
    if not gated:
        return
    ruta_env = raiz / ".env"
    nombres = ", ".join(gated)
    if not ruta_env.exists():
        raise ValueError(
            f"La corrida incluye modelos gated ({nombres}) pero no existe el "
            f"archivo {ruta_env}. Creá .env en la raíz del repo con "
            f"{CLAVE_TOKEN}=<tu token> (ver docker/README.md)."
        )
    if not _hf_token_de_env(ruta_env):
        raise ValueError(
            f"La corrida incluye modelos gated ({nombres}) pero {CLAVE_TOKEN} "
            f"está ausente o vacío en {ruta_env}. Completalo "
            f"(ver docker/README.md)."
        )


# --------------------------------------------------------------------------
# Orquestación secuencial y reanudable
# --------------------------------------------------------------------------

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


def _correr_modelo(modelo: ModeloEvaluado2026, posicion: str,
                   force: bool, dry_run: bool) -> int:
    """Corre el contenedor de un modelo y devuelve su código de salida."""
    cmd = comando_run(modelo, RAIZ, force)
    print(f"\n=== [{posicion}] {modelo.nombre} ({modelo.tier}) ===")
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def ejecutar_barrido(modelos: list[ModeloEvaluado2026],
                     force: bool, dry_run: bool) -> int:
    """De a un contenedor por vez, en orden de roster; corta al primer fallo.

    Cortar es lo correcto: los modelos ya terminados quedan en disco y el
    harness los saltea, así que relanzar no rehace nada (RF3).
    """
    for i, modelo in enumerate(modelos, 1):
        codigo = _correr_modelo(modelo, f"{i}/{len(modelos)}", force, dry_run)
        if codigo != 0:
            print(f"FALLO {modelo.nombre} (código {codigo}). "
                  f"Corregí y retomá con --desde \"{modelo.nombre}\"; los "
                  f"modelos ya completos se saltean solos.", file=sys.stderr)
            return codigo
    print("\nBarrido completo.")
    return 0


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido 2026: las 12 imágenes del roster activo, de a una"
    )
    parser.add_argument("--desde", default=None,
                        help="retomar desde este nombre de modelo")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV del modelo ya esté completo")
    parser.add_argument("--dry-run", action="store_true",
                        help="imprimir los comandos sin ejecutarlos")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    modelos = seleccionar_modelos(args.desde)
    validar_credenciales(modelos, RAIZ)  # F11: antes de correr nada
    return ejecutar_barrido(modelos, args.force, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
