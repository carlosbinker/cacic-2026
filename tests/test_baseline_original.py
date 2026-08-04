"""Tests de la corrida de BASELINE-COMPLETION (Delta 05, subtask 19).

Motivo: un control con el prompt ORIGINAL del paper (subtask 18) NO reprodujo
la Tabla 2 publicada -- la causa, con evidencia, es que el entorno de la
corrida original (versión de `transformers`) nunca quedó fijado ni
registrado (F0 no tiene columna `transformers_version`), así que es
irreproducible, no un defecto del harness nuevo. Las cifras publicadas pasan
a referencia histórica; el paper compara en cambio contra un baseline
RE-MEDIDO de los 4 modelos del paper original bajo el harness/prompt 2026.

Tres de esos 4 (`SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`,
`Qwen2.5-1.5B-Instruct`) ya corren en el barrido principal
(`data/2026/detalle/`, `activo=True`). Esta corrida cubre el cuarto,
`Qwen2.5-0.5B-Instruct` (`baseline_original=True`, `activo=False`), bajo el
prompt 2026 -- NO el de control -- reusando `run_sweep_2026.evaluar_modelo`
tal cual (sin builder alternativo: su default ya es el prompt 2026).

Ninguno de estos tests ejerce inferencia real ni escribe bajo `data/2026/`:
todo el I/O pasa por `tmp_path`/`monkeypatch`, siguiendo la misma convención
que `tests/test_control_prompt_original.py` y `tests/test_docker_matriz.py`.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))

from models_2026 import por_nombre, roster_baseline_original, slug  # noqa: E402
from prompt_2026 import SYSTEM_PROMPT_2026  # noqa: E402

import run_sweep_2026  # noqa: E402
import run_baseline_original as rbo  # noqa: E402
import run_baseline  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Task 3: src/run_baseline_original.py
# --------------------------------------------------------------------------

def test_modelos_baseline_completion_es_solo_el_que_falta():
    """Los otros 3 baseline_original ya estan activos: no se re-corren aca."""
    nombres = [m.nombre for m in rbo.modelos_baseline_completion()]
    assert nombres == ["Qwen2.5-0.5B-Instruct"]


def test_modelos_baseline_completion_es_subconjunto_de_roster_baseline_original():
    completion = {m.nombre for m in rbo.modelos_baseline_completion()}
    baseline = {m.nombre for m in roster_baseline_original()}
    assert completion <= baseline


def test_ruta_baseline_usa_el_slug_bajo_baseline_original():
    ruta = rbo.ruta_baseline("Qwen2.5-0.5B-Instruct")
    assert ruta.name == f"{slug('Qwen2.5-0.5B-Instruct')}.csv"
    assert ruta.parent.name == "baseline_original"
    assert ruta.parent.parent.name == "2026"


def test_verificar_es_baseline_completion_rechaza_un_modelo_fuera_del_conjunto():
    otro = por_nombre("LFM2.5-230M")
    with pytest.raises(ValueError, match="no es un modelo de baseline-completion"):
        rbo.verificar_es_baseline_completion(otro)


def test_verificar_es_baseline_completion_rechaza_un_anclaje_ya_activo():
    """Qwen2.5-1.5B-Instruct es baseline_original PERO ya esta activo: no es
    baseline-completion, correrlo aca duplicaria data/2026/detalle/."""
    ancla = por_nombre("Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="no es un modelo de baseline-completion"):
        rbo.verificar_es_baseline_completion(ancla)


def test_verificar_es_baseline_completion_acepta_el_modelo_correcto():
    rbo.verificar_es_baseline_completion(por_nombre("Qwen2.5-0.5B-Instruct"))  # no debe lanzar


def test_evaluar_modelo_usa_el_prompt_2026_por_defecto_no_el_de_control(monkeypatch):
    """A diferencia de run_control_prompt_original, esta corrida NO sustituye el
    prompt de sistema: reusa evaluar_modelo sin pasar construir_entrada."""

    class TokenizerConChat:
        chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

        def __init__(self):
            self.mensajes_recibidos = None

        def apply_chat_template(self, mensajes, **kwargs):
            self.mensajes_recibidos = mensajes
            return {"input_ids": [[1, 2, 3]]}

    class FakeCtx:
        modelo = por_nombre("Qwen2.5-0.5B-Instruct")
        tokenizer = TokenizerConChat()
        transformers_version = "4.57.6"

    capturado = {}

    def fake_abrir_contexto(_modelo):
        return FakeCtx()

    def fake_generar_respuesta(ctx, entrada):
        capturado["mensajes"] = ctx.tokenizer.mensajes_recibidos
        return (
            '{"intent": "encender", "dispositivo": "luz", "ubicacion": "living", '
            '"valor": null, "unidad": null}',
            0.01,
        )

    monkeypatch.setattr(run_sweep_2026, "_abrir_contexto", fake_abrir_contexto)
    monkeypatch.setattr(run_sweep_2026, "_generar_respuesta", fake_generar_respuesta)

    dataset = run_sweep_2026.cargar_dataset().head(1)
    run_sweep_2026.evaluar_modelo(por_nombre("Qwen2.5-0.5B-Instruct"), dataset)
    assert capturado["mensajes"][0]["content"] == SYSTEM_PROMPT_2026


def test_main_escribe_el_csv_de_baseline_con_las_columnas_congeladas(monkeypatch, tmp_path):
    ruta_tmp = tmp_path / "qwen2-5-0-5b-instruct.csv"
    monkeypatch.setattr(rbo, "ruta_baseline", lambda nombre: ruta_tmp)

    dataset = run_sweep_2026.cargar_dataset()

    def fake_evaluar_modelo(modelo_, dataset_):
        filas = []
        for idx, fila_ds in dataset_.iterrows():
            filas.append(run_sweep_2026.armar_fila(
                modelo=modelo_, idx=int(idx), comando=fila_ds["comando"],
                gt={"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
                    "valor": None, "unidad": None},
                texto_generado='{"intent": "encender", "dispositivo": "luz", '
                               '"ubicacion": "living", "valor": null, "unidad": null}',
                latencia_s=0.01, modo="chat_template", transformers_version="4.57.6",
            ))
        return filas

    monkeypatch.setattr(rbo, "evaluar_modelo", fake_evaluar_modelo)
    monkeypatch.setattr(rbo, "cargar_dataset", lambda: dataset)
    monkeypatch.setattr(sys, "argv", [
        "run_baseline_original.py", "--modelo", "Qwen2.5-0.5B-Instruct",
    ])

    codigo = rbo.main()
    assert codigo == 0
    df = pd.read_csv(ruta_tmp)
    assert list(df.columns) == run_sweep_2026.COLUMNAS_DETALLE
    assert len(df) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert sorted(df["idx"]) == list(range(run_sweep_2026.N_COMANDOS_ESPERADO))


def test_main_rechaza_un_modelo_fuera_del_conjunto_de_baseline_completion(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_baseline_original.py", "--modelo", "LFM2.5-230M",
    ])
    with pytest.raises(ValueError, match="no es un modelo de baseline-completion"):
        rbo.main()


def test_main_saltea_si_el_csv_de_baseline_ya_esta_completo(monkeypatch, tmp_path):
    ruta_tmp = tmp_path / "ya-completo.csv"
    pd.DataFrame(
        {c: [None] * run_sweep_2026.N_COMANDOS_ESPERADO for c in run_sweep_2026.COLUMNAS_DETALLE}
        | {"idx": list(range(run_sweep_2026.N_COMANDOS_ESPERADO))}
    ).to_csv(ruta_tmp, index=False)

    monkeypatch.setattr(rbo, "ruta_baseline", lambda nombre: ruta_tmp)

    def fallar_si_se_llama(*_a, **_k):
        raise AssertionError("no deberia evaluarse: el CSV ya esta completo")

    monkeypatch.setattr(rbo, "evaluar_modelo", fallar_si_se_llama)
    monkeypatch.setattr(sys, "argv", [
        "run_baseline_original.py", "--modelo", "Qwen2.5-0.5B-Instruct",
    ])
    assert rbo.main() == 0


def test_el_modulo_de_baseline_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "run_baseline_original.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea


# --------------------------------------------------------------------------
# Task 3: docker/run_baseline.py
# --------------------------------------------------------------------------

def test_modelos_baseline_es_el_mismo_conjunto_que_run_baseline_original():
    assert [m.nombre for m in run_baseline.modelos_baseline()] == \
        [m.nombre for m in rbo.modelos_baseline_completion()]


def test_comando_run_baseline_respeta_el_mismo_envolvente_de_recursos():
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    cmd = run_baseline.comando_run_baseline(modelo, RAIZ)
    assert "--memory=8g" in cmd and "--cpus=2" in cmd
    assert f"--cpuset-cpus={run_baseline.CPUSET_POR_DEFECTO}" in cmd
    assert run_baseline.CPUSET_POR_DEFECTO == "0-1"
    assert "--rm" in cmd


def test_comando_run_baseline_monta_datos_cache_y_src_de_solo_lectura():
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    cmd = run_baseline.comando_run_baseline(modelo, RAIZ)
    montajes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert any(m.endswith(":/app/data") for m in montajes)
    assert any(m.endswith(":/app/.hf_cache") for m in montajes)
    assert any(m.endswith(":/app/src:ro") for m in montajes), (
        "la imagen hornea src/ al build time; sin este montaje el entrypoint nuevo "
        "es invisible dentro del contenedor (mismo error que ya costo una corrida fallida)"
    )


def test_comando_run_baseline_no_usa_env_file_no_es_gated():
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    assert modelo.gated is False
    cmd = run_baseline.comando_run_baseline(modelo, RAIZ)
    assert "--env-file" not in cmd


def test_comando_run_baseline_invoca_el_entrypoint_de_baseline():
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    cmd = run_baseline.comando_run_baseline(modelo, RAIZ)
    assert cmd[-3:] == ["src/run_baseline_original.py", "--modelo", modelo.nombre]


def test_comando_run_baseline_usa_la_imagen_del_modelo():
    from build_all import tag_imagen
    modelo = por_nombre("Qwen2.5-0.5B-Instruct")
    cmd = run_baseline.comando_run_baseline(modelo, RAIZ)
    assert tag_imagen(modelo) in cmd
    assert tag_imagen(modelo) == "slm-domotica-2026:qwen2-5-0-5b-instruct"
    assert cmd[:2] == ["docker", "run"]


def test_dry_run_imprime_un_comando_con_el_envolvente_completo(capsys):
    codigo = run_baseline.ejecutar_baseline(
        run_baseline.modelos_baseline(), force=False, dry_run=True
    )
    assert codigo == 0
    lineas_comando = [
        linea for linea in capsys.readouterr().out.splitlines() if linea.startswith("docker run")
    ]
    assert len(lineas_comando) == 1
    linea = lineas_comando[0]
    assert "--memory=8g" in linea and "--cpus=2" in linea and "--cpuset-cpus=0-1" in linea
    assert ":/app/src:ro" in linea


def test_ejecutar_baseline_no_corta_al_primer_fallo_y_registra_verbatim(monkeypatch, tmp_path):
    corridos = []

    def falso_correr_modelo(modelo, _posicion, _force, _dry_run, _cpuset):
        corridos.append(modelo.nombre)
        return 1, "boom textual"

    monkeypatch.setattr(run_baseline, "_correr_modelo", falso_correr_modelo)
    monkeypatch.setattr(run_baseline, "_csv_ya_completo", lambda _ruta: False)
    monkeypatch.setattr(run_baseline, "FALLOS_PATH", tmp_path / "fallos_baseline.json")

    codigo = run_baseline.ejecutar_baseline(
        run_baseline.modelos_baseline(), force=False, dry_run=False
    )
    assert codigo == 1
    assert corridos == [m.nombre for m in run_baseline.modelos_baseline()]

    fallos = json.loads((tmp_path / "fallos_baseline.json").read_text(encoding="utf-8"))
    assert len(fallos) == 1
    assert fallos[0]["error_textual"] == "boom textual"
    assert set(fallos[0]) == {
        "modelo", "hf_repo_id", "transformers_pin", "codigo_salida", "error_textual", "momento_iso",
    }


def test_ejecutar_baseline_sin_fallos_escribe_lista_vacia_siempre(monkeypatch, tmp_path):
    monkeypatch.setattr(run_baseline, "_correr_modelo", lambda *_a, **_k: (0, ""))
    monkeypatch.setattr(run_baseline, "_csv_ya_completo", lambda _ruta: False)
    monkeypatch.setattr(run_baseline, "FALLOS_PATH", tmp_path / "fallos_baseline.json")
    assert run_baseline.ejecutar_baseline(
        run_baseline.modelos_baseline(), force=False, dry_run=False
    ) == 0
    assert json.loads((tmp_path / "fallos_baseline.json").read_text(encoding="utf-8")) == []


def test_dry_run_no_escribe_fallos_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(run_baseline, "FALLOS_PATH", tmp_path / "fallos_baseline.json")
    assert run_baseline.ejecutar_baseline(
        run_baseline.modelos_baseline(), force=False, dry_run=True
    ) == 0
    assert not (tmp_path / "fallos_baseline.json").exists()


def test_ejecutar_baseline_saltea_si_el_csv_ya_esta_completo(monkeypatch, tmp_path):
    llamados = []
    monkeypatch.setattr(run_baseline, "FALLOS_PATH", tmp_path / "fallos_baseline.json")
    monkeypatch.setattr(run_baseline, "_csv_ya_completo", lambda _ruta: True)

    def falso_correr_modelo(modelo, *_a, **_k):
        llamados.append(modelo.nombre)
        return 0, ""

    monkeypatch.setattr(run_baseline, "_correr_modelo", falso_correr_modelo)
    codigo = run_baseline.ejecutar_baseline(
        run_baseline.modelos_baseline(), force=False, dry_run=False
    )
    assert codigo == 0
    assert llamados == []


def test_no_hay_credenciales_en_run_baseline():
    import re
    fuente = (RAIZ / "docker" / "run_baseline.py").read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", fuente)
    assert "ARG HF_TOKEN" not in fuente and "ENV HF_TOKEN" not in fuente
