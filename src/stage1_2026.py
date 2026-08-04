#!/usr/bin/env python3
"""
Cierre de la etapa 1: consolida los CSV por modelo, resume, y elige el juez.

La selección del juez es una función pura y total del resumen (RF7), para que
dos corridas sobre los mismos datos elijan siempre el mismo modelo. La
reproducibilidad de la etapa 2 la garantiza la decodificación greedy, no este
archivo; registrar al ganador es trazabilidad.

Uso:
    python src/stage1_2026.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from models_2026 import MODELOS_2026, por_nombre, roster_activo, slug
from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
DIR_DETALLE = DIR_2026 / "detalle"
PATH_CONSOLIDADO = DIR_2026 / "detalle_2026.csv"
PATH_RESUMEN_E1 = DIR_2026 / "resumen_etapa1.json"
PATH_JUEZ = DIR_2026 / "juez_seleccionado.json"


def consolidar_detalle(dir_detalle: Path, nombres: list[str]) -> pd.DataFrame:
    """Concatena los CSV por modelo en orden de roster; falla si falta alguno."""
    partes, faltantes = [], []
    for nombre in nombres:
        ruta = dir_detalle / f"{slug(nombre)}.csv"
        if not ruta.exists():
            ruta = dir_detalle / f"{nombre}.csv"   # tolerancia en tests
        if not ruta.exists():
            faltantes.append(nombre)
            continue
        df = pd.read_csv(ruta)
        if list(df.columns) != COLUMNAS_DETALLE:
            raise ValueError(f"{ruta} tiene un esquema distinto al congelado en F5")
        if len(df) != N_COMANDOS_ESPERADO:
            raise ValueError(f"{ruta} tiene {len(df)} filas, se esperaban {N_COMANDOS_ESPERADO}")
        partes.append(df.sort_values("idx"))
    if faltantes:
        raise ValueError(f"faltan CSV de detalle para: {', '.join(faltantes)}")
    return pd.concat(partes, ignore_index=True)[COLUMNAS_DETALLE]


def resumen_etapa1(df: pd.DataFrame, params_por_modelo: dict[str, float]) -> list[dict]:
    """Resumen previo a la etapa 2: sin métrica laxa, que aún no existe."""
    filas = []
    for nombre, params_b in params_por_modelo.items():
        df_m = df[df["modelo"] == nombre]
        if df_m.empty:
            continue
        filas.append({
            "modelo": nombre,
            "params_b": params_b,
            "n": len(df_m),
            "json_valido_pct": round(100 * df_m["json_valido"].mean(), 1),
            "exact_match_pct": round(100 * df_m["match_exact"].mean(), 1),
            "avg_latencia_s": round(df_m["latencia_s"].mean(), 3),
        })
    return filas


N_ROSTER_ACTIVO = 12
PATH_FALLOS = DIR_2026 / "fallos_barrido.json"


def calcular_filas_esperadas(fallos: list[dict], n_roster: int = N_ROSTER_ACTIVO,
                              n_comandos: int = N_COMANDOS_ESPERADO) -> int:
    """F12.4: cada modelo fallido en el barrido resta sus 32 filas del total."""
    return (n_roster - len(fallos)) * n_comandos


def nombres_disponibles(nombres_roster: list[str], fallos: list[dict]) -> list[str]:
    """Roster (activo) menos los modelos que fallaron el barrido (F12.4)."""
    fallidos = {f["modelo"] for f in fallos}
    return [n for n in nombres_roster if n not in fallidos]


def reportar_fallos(fallos: list[dict]) -> None:
    """F12.4: nunca deja pasar en silencio un consolidado corto."""
    if not fallos:
        return
    print(f"\nATENCION: {len(fallos)} modelo(s) fuera del consolidado (F12.4):")
    for f in fallos:
        print(f"  {f['modelo']} (codigo_salida={f['codigo_salida']}): {f['error_textual']}")
    print("  Quedan excluidos de toda tabla/figura y deben reportarse como limitacion "
          "en el paper (subtasks 14/15).")


CRITERIO_JUEZ = (
    "mayor exact_match_pct en etapa 1; desempate por mayor params_b; "
    "luego orden del roster"
)


def elegir_juez(resumen: list[dict]) -> dict:
    """Elige el juez de la etapa 2 (RF7). Función pura y total del resumen."""
    if not resumen:
        raise ValueError("El resumen de etapa 1 está vacío: no hay juez posible")

    orden_roster = {m.nombre: i for i, m in enumerate(MODELOS_2026)}
    mejor_pct = max(f["exact_match_pct"] for f in resumen)
    empatados = [f for f in resumen if f["exact_match_pct"] == mejor_pct]
    ganador = min(
        empatados,
        key=lambda f: (-f["params_b"], orden_roster.get(f["modelo"], len(orden_roster))),
    )
    try:
        hf_repo_id = por_nombre(ganador["modelo"]).hf_repo_id
    except ValueError:
        # Fuera del roster real: solo ocurre con resúmenes sintéticos de test:
        # la selección en sí (RF7) no depende de que el modelo exista en el
        # registro, solo el enriquecimiento con hf_repo_id lo requiere.
        hf_repo_id = ""
    return {
        "modelo": ganador["modelo"],
        "hf_repo_id": hf_repo_id,
        "params_b": ganador["params_b"],
        "exact_match_pct": ganador["exact_match_pct"],
        "criterio": CRITERIO_JUEZ,
        "empatados": sorted(f["modelo"] for f in empatados),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cierre de la etapa 1")
    parser.add_argument("--dir-detalle", default=str(DIR_DETALLE))
    args = parser.parse_args()

    fallos = (
        json.loads(PATH_FALLOS.read_text(encoding="utf-8")) if PATH_FALLOS.exists() else []
    )
    reportar_fallos(fallos)
    nombres = nombres_disponibles([m.nombre for m in roster_activo()], fallos)

    df = consolidar_detalle(Path(args.dir_detalle), nombres)
    esperadas = calcular_filas_esperadas(fallos)
    if len(df) != esperadas:
        raise ValueError(
            f"consolidado con {len(df)} filas, se esperaban {esperadas} "
            f"({N_ROSTER_ACTIVO - len(fallos)} modelos activos x {N_COMANDOS_ESPERADO} comandos)"
        )
    PATH_CONSOLIDADO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_CONSOLIDADO, index=False)

    params_por_modelo = {m.nombre: m.params_b for m in roster_activo() if m.nombre in nombres}
    resumen = resumen_etapa1(df, params_por_modelo)
    PATH_RESUMEN_E1.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    juez = elegir_juez(resumen)
    PATH_JUEZ.write_text(json.dumps(juez, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Consolidado: {len(df)} filas ({len(nombres)} modelos, "
          f"{len(fallos)} excluidos por fallo de barrido) en {PATH_CONSOLIDADO}")
    for f in sorted(resumen, key=lambda x: -x["exact_match_pct"]):
        print(f"  {f['modelo']:26s} exact={f['exact_match_pct']:5.1f}%  "
              f"lat={f['avg_latencia_s']:7.2f}s")
    print(f"\nJuez elegido: {juez['modelo']} ({juez['exact_match_pct']}%), "
          f"empatados: {juez['empatados']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


