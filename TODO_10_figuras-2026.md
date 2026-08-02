---
id: 10
title: Figuras horizontales agrupadas por tier
depends_on: [09]
files:
  - src/generate_figures_2026.py
  - tests/test_generate_figures_2026.py
  - figures/2026/
---

## Spec

Implementar **F7** / **RF12**: rediseñar las figuras para que soporten 12 modelos. Las actuales son barras verticales para 4 modelos; a 12, con nombres como `granite-4.0-h-350m` y `OLMo-2-0425-1B-Instruct`, se vuelven ilegibles.

Diseño: **barras horizontales**, agrupadas y separadas visualmente por `tier` (`sub-1B` arriba, `1-2B` abajo), con los nombres completos como etiquetas del eje Y — que es precisamente lo que la orientación horizontal permite sin rotar texto.

Módulo **nuevo**. `src/generate_figures.py` queda congelado por **F0** para que las figuras publicadas del paper original sigan siendo reproducibles bit a bit; conserva de él la convención de CLI (`--resumen`, `--sufijo`) y agrega `--salida-dir`.

Fig. 1 muestra **ambas** exactitudes (estricta y laxa) en el panel izquierdo — la brecha es el resultado central — y la latencia promedio en el derecho. Fig. 2 muestra exactitud por campo del esquema.

## Implementation plan

### Tarea 1 — Preparación de datos para graficar (TDD)

- [ ] Escribir el test que falla, `tests/test_generate_figures_2026.py`:

```python
"""Tests de las figuras 2026: preparación de datos y generación de archivos."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from generate_figures_2026 import (  # noqa: E402
    CAMPOS_FIG2,
    generar_figuras,
    ordenar_para_grafico,
    posiciones_con_separacion,
)
from models_2026 import MODELOS_2026  # noqa: E402


def _resumen():
    filas = []
    for i, m in enumerate(MODELOS_2026):
        filas.append({
            "modelo": m.nombre, "hf_repo_id": m.hf_repo_id, "params_b": m.params_b,
            "tier": m.tier, "modo_prompting": "chat_template",
            "transformers_pin": m.transformers_pin, "n": 32,
            "json_valido_pct": 100.0,
            "exact_match_pct": float(10 + i * 4),
            "exact_match_laxo_pct": float(15 + i * 4),
            "avg_latencia_s": float(5 + i * 3),
            "acc_intent_pct": 80.0, "acc_dispositivo_pct": 70.0,
            "acc_ubicacion_pct": 75.0, "acc_valor_pct": 85.0, "acc_unidad_pct": 90.0,
        })
    return filas


def test_ordena_sub1b_primero_y_por_params_dentro_del_tier():
    ordenado = ordenar_para_grafico(_resumen())
    tiers = [f["tier"] for f in ordenado]
    assert tiers == ["sub-1B"] * 6 + ["1-2B"] * 6
    sub = [f["params_b"] for f in ordenado[:6]]
    grandes = [f["params_b"] for f in ordenado[6:]]
    assert sub == sorted(sub) and grandes == sorted(grandes)


def test_ordenar_no_pierde_ni_duplica_modelos():
    ordenado = ordenar_para_grafico(_resumen())
    assert len(ordenado) == 12
    assert {f["modelo"] for f in ordenado} == {m.nombre for m in MODELOS_2026}


def test_hay_un_hueco_entre_los_dos_tiers():
    tiers = ["sub-1B"] * 6 + ["1-2B"] * 6
    pos = posiciones_con_separacion(tiers)
    assert len(pos) == 12
    assert all(b > a for a, b in zip(pos, pos[1:]))          # monótono
    dentro = pos[1] - pos[0]
    entre = pos[6] - pos[5]
    assert entre > dentro, (entre, dentro)                   # hueco entre tiers


def test_posiciones_con_un_solo_tier_no_deja_hueco():
    pos = posiciones_con_separacion(["sub-1B"] * 3)
    assert [round(b - a, 6) for a, b in zip(pos, pos[1:])] == [1.0, 1.0]


def test_campos_fig2_son_los_cinco_del_esquema():
    assert CAMPOS_FIG2 == [
        ("acc_intent_pct", "intención"),
        ("acc_dispositivo_pct", "dispositivo"),
        ("acc_ubicacion_pct", "ubicación"),
        ("acc_valor_pct", "valor"),
        ("acc_unidad_pct", "unidad"),
    ]


def test_generar_figuras_escribe_los_dos_png(tmp_path):
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(_resumen(), ensure_ascii=False), encoding="utf-8")
    salidas = generar_figuras(resumen, tmp_path / "figs", sufijo="")
    assert len(salidas) == 2
    for ruta in salidas:
        assert ruta.exists() and ruta.stat().st_size > 5000
    assert salidas[0].name == "fig1_exactitud_latencia_2026.png"
    assert salidas[1].name == "fig2_exactitud_por_campo_2026.png"


def test_el_sufijo_no_pisa_las_figuras_oficiales(tmp_path):
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(_resumen(), ensure_ascii=False), encoding="utf-8")
    salidas = generar_figuras(resumen, tmp_path / "figs", sufijo="_reproducido")
    assert salidas[0].name == "fig1_exactitud_latencia_2026_reproducido.png"


def test_generar_figuras_falla_si_falta_la_metrica_laxa(tmp_path):
    datos = _resumen()
    for f in datos:
        del f["exact_match_laxo_pct"]
    resumen = tmp_path / "r.json"
    resumen.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact_match_laxo_pct"):
        generar_figuras(resumen, tmp_path / "figs", sufijo="")
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_generate_figures_2026.py`
- [ ] Implementar `src/generate_figures_2026.py`:

```python
#!/usr/bin/env python3
"""
Figuras del estudio 2026.

Módulo nuevo, no una edición de generate_figures.py: aquel está congelado
para que las figuras publicadas del paper original sigan siendo reproducibles.
Conserva su convención de CLI (--resumen / --sufijo).

Rediseño: barras horizontales agrupadas por tier. Con 12 modelos y nombres
como 'OLMo-2-0425-1B-Instruct', las barras verticales del diseño original
obligan a rotar las etiquetas hasta volverlas ilegibles; en horizontal el
nombre completo entra en el eje Y.

Uso:
    python src/generate_figures_2026.py [--resumen R] [--sufijo S] [--salida-dir D]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PATH_RESUMEN = RAIZ / "data" / "2026" / "resumen_2026.json"
DIR_FIGURAS = RAIZ / "figures" / "2026"

ORDEN_TIERS = ["sub-1B", "1-2B"]
SEPARACION_ENTRE_TIERS = 1.0   # en unidades de barra

CAMPOS_FIG2 = [
    ("acc_intent_pct", "intención"),
    ("acc_dispositivo_pct", "dispositivo"),
    ("acc_ubicacion_pct", "ubicación"),
    ("acc_valor_pct", "valor"),
    ("acc_unidad_pct", "unidad"),
]


def ordenar_para_grafico(resumen: list[dict]) -> list[dict]:
    """sub-1B primero, y dentro de cada tier por tamaño creciente."""
    return sorted(
        resumen,
        key=lambda f: (ORDEN_TIERS.index(f["tier"]), f["params_b"], f["modelo"]),
    )


def posiciones_con_separacion(tiers: list[str]) -> list[float]:
    """Posiciones en el eje Y, con un hueco visual al cambiar de tier."""
    posiciones, actual = [], 0.0
    for i, tier in enumerate(tiers):
        if i > 0 and tier != tiers[i - 1]:
            actual += SEPARACION_ENTRE_TIERS
        posiciones.append(actual)
        actual += 1.0
    return posiciones


def _validar(resumen: list[dict]) -> None:
    if not resumen:
        raise ValueError("El resumen está vacío: no hay nada que graficar")
    requeridas = {"modelo", "tier", "params_b", "exact_match_pct",
                  "exact_match_laxo_pct", "avg_latencia_s"}
    faltan = requeridas - set(resumen[0])
    if faltan:
        raise ValueError(f"Al resumen le faltan claves: {sorted(faltan)}")


def _etiquetas_de_tier(ax, tiers, posiciones):
    """Anota el nombre del tier a la altura de su primer modelo."""
    for tier in ORDEN_TIERS:
        indices = [i for i, t in enumerate(tiers) if t == tier]
        if indices:
            ax.text(-0.02, posiciones[indices[0]] - 0.7, tier,
                    transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=9, fontweight="bold", color="#444444")


def generar_figuras(path_resumen: Path, dir_salida: Path, sufijo: str) -> list[Path]:
    resumen = ordenar_para_grafico(
        json.loads(Path(path_resumen).read_text(encoding="utf-8"))
    )
    _validar(resumen)

    nombres = [f["modelo"] for f in resumen]
    tiers = [f["tier"] for f in resumen]
    pos = posiciones_con_separacion(tiers)
    dir_salida.mkdir(parents=True, exist_ok=True)
    salidas = []

    # --- Fig. 1: exactitud (estricta vs laxa) | latencia ---
    alto = max(5.0, 0.42 * (max(pos) + 2))
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, alto))

    izq.barh([p - 0.2 for p in pos], [f["exact_match_pct"] for f in resumen],
             height=0.4, label="estricta", color="#2b6cb0")
    izq.barh([p + 0.2 for p in pos], [f["exact_match_laxo_pct"] for f in resumen],
             height=0.4, label="laxa (semántica)", color="#90cdf4")
    izq.set_yticks(pos, nombres, fontsize=9)
    izq.invert_yaxis()
    izq.set_xlabel("Coincidencia exacta (%)")
    izq.set_xlim(0, 100)
    izq.legend(loc="lower right", fontsize=9)
    izq.grid(axis="x", alpha=0.3)
    _etiquetas_de_tier(izq, tiers, pos)

    der.barh(pos, [f["avg_latencia_s"] for f in resumen], height=0.6, color="#dd6b20")
    der.set_yticks(pos, ["" for _ in nombres])
    der.invert_yaxis()
    der.set_xlabel("Latencia promedio en CPU (s)")
    der.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    ruta1 = dir_salida / f"fig1_exactitud_latencia_2026{sufijo}.png"
    fig.savefig(ruta1, dpi=200)
    plt.close(fig)
    salidas.append(ruta1)

    # --- Fig. 2: exactitud por campo del esquema ---
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.55 * (max(pos) + 2))))
    n = len(CAMPOS_FIG2)
    alto_barra = 0.8 / n
    for k, (clave, etiqueta) in enumerate(CAMPOS_FIG2):
        desplazamiento = (k - (n - 1) / 2) * alto_barra
        ax.barh([p + desplazamiento for p in pos],
                [f[clave] for f in resumen], height=alto_barra, label=etiqueta)
    ax.set_yticks(pos, nombres, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Exactitud por campo (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    _etiquetas_de_tier(ax, tiers, pos)

    fig.tight_layout()
    ruta2 = dir_salida / f"fig2_exactitud_por_campo_2026{sufijo}.png"
    fig.savefig(ruta2, dpi=200)
    plt.close(fig)
    salidas.append(ruta2)

    return salidas


def main() -> int:
    parser = argparse.ArgumentParser(description="Figuras del estudio 2026")
    parser.add_argument("--resumen", default=str(PATH_RESUMEN))
    parser.add_argument("--sufijo", default="")
    parser.add_argument("--salida-dir", default=str(DIR_FIGURAS))
    args = parser.parse_args()

    for ruta in generar_figuras(Path(args.resumen), Path(args.salida_dir), args.sufijo):
        print(f"escrita: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_generate_figures_2026.py`

### Tarea 2 — Generar las figuras reales y revisarlas

- [ ] `python src/generate_figures_2026.py`
- [ ] Abrir los dos PNG y confirmar visualmente: los 12 nombres se leen completos y sin recortes, los dos tiers están separados y rotulados, ninguna barra sale del área, y en fig. 1 se distingue la barra estricta de la laxa.
- [ ] Si algún nombre queda cortado, ajustar `figsize`/`tight_layout`, no acortar los nombres (deben coincidir con los de la Tabla 1).

### Tarea 3 — commit

- [ ] `pytest -q`
- [ ] `git add src/generate_figures_2026.py tests/test_generate_figures_2026.py figures/2026/`
- [ ] `git commit -m "feat(2026): figuras horizontales agrupadas por tier con exactitud estricta y laxa"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Las dos figuras existen y no están vacías
ls -l figures/2026/fig1_exactitud_latencia_2026.png figures/2026/fig2_exactitud_por_campo_2026.png
python -c "
from pathlib import Path
for n in ('fig1_exactitud_latencia_2026.png','fig2_exactitud_por_campo_2026.png'):
    p = Path('figures/2026')/n
    assert p.exists() and p.stat().st_size > 20000, (n, p.stat().st_size)
print('figuras OK')
"

# 3. Las figuras publicadas del paper original NO fueron tocadas
git diff --exit-code main -- figures/fig1_exactitud_latencia.png \
  figures/fig2_exactitud_por_campo.png src/generate_figures.py \
  && echo "figuras y script legacy intactos"

# 4. El sufijo permite reproducir sin pisar
python src/generate_figures_2026.py --sufijo _reproducido
ls figures/2026/ | grep _reproducido
rm figures/2026/*_reproducido.png

# 5. Los 12 modelos aparecen en el gráfico (chequeo por conteo de ticks)
python -c "
import sys, json; sys.path.insert(0,'src')
from generate_figures_2026 import ordenar_para_grafico, posiciones_con_separacion
r = ordenar_para_grafico(json.load(open('data/2026/resumen_2026.json', encoding='utf-8')))
assert len(r) == 12
tiers = [f['tier'] for f in r]
assert tiers == ['sub-1B']*6 + ['1-2B']*6, tiers
pos = posiciones_con_separacion(tiers)
assert pos[6]-pos[5] > pos[1]-pos[0]
print('orden y separacion por tier OK')
"
```

## Acceptance criteria

- **Dado** el resumen de 12 modelos, **cuando** se llama `ordenar_para_grafico`, **entonces** devuelve los 6 `sub-1B` primero y los 6 `1-2B` después, ordenados por `params_b` creciente dentro de cada tier, sin perder ni duplicar modelos.
- **Dado** una lista de tiers, **cuando** se llama `posiciones_con_separacion`, **entonces** las posiciones son estrictamente crecientes y el salto en el cambio de tier es mayor que el salto dentro de un tier; con un solo tier, todos los saltos valen 1.0.
- **Dado** `python src/generate_figures_2026.py`, **entonces** escribe `figures/2026/fig1_exactitud_latencia_2026.png` y `figures/2026/fig2_exactitud_por_campo_2026.png`, ambos de más de 20 KB.
- **Dado** la figura 1, **entonces** su panel izquierdo muestra **dos** series por modelo (exactitud estricta y laxa) con leyenda, y el derecho la latencia promedio; ambos con barras **horizontales** y los 12 nombres completos legibles en el eje Y.
- **Dado** la figura 2, **entonces** muestra los 5 campos del esquema (`intención`, `dispositivo`, `ubicación`, `valor`, `unidad`) como series agrupadas por modelo.
- **Dado** ambas figuras, **entonces** los dos tiers están visualmente separados y rotulados (`sub-1B`, `1-2B`).
- **Dado** `--sufijo _reproducido`, **entonces** los archivos se escriben con ese sufijo y no pisan a los oficiales.
- **Dado** un resumen sin la clave `exact_match_laxo_pct`, **entonces** `generar_figuras` lanza `ValueError` nombrando la clave faltante en vez de dibujar una figura incompleta.
- **Dado** `git diff main`, **entonces** `src/generate_figures.py` y los dos PNG publicados en `figures/` siguen byte-idénticos.
