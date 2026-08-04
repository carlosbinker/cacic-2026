"""Tests de la corrida de CONTROL (Delta 04): mismo harness 2026, mismo
dataset, mismo orden, misma decodificación y mismo scoring que el barrido
principal -- la ÚNICA variable que cambia es el prompt de sistema, que pasa
de `SYSTEM_PROMPT_2026` a `SYSTEM_PROMPT_PAPER` (el original del paper,
importado tal cual de `src/prompt.py`, F0 congelado).

Ninguno de estos tests ejerce inferencia real (no se descarga ni carga
ningún modelo) ni escribe bajo `data/2026/`: todo el I/O de archivos pasa
por `tmp_path`, siguiendo la misma convención de
`tests/test_run_sweep_2026.py` y `tests/test_docker_matriz.py`.

Este archivo crece en tres etapas (Delta 04, tasks 1-3): primero la
infraestructura factorizada y el entrypoint de control (`src/
run_control_prompt_original.py`), luego el runner host-side
(`docker/run_control.py`) y por último el generador de comparación de tres
bandas (`src/comparar_control.py`).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker"))

from models_2026 import por_nombre, slug  # noqa: E402
from prompt import SYSTEM_PROMPT_PAPER  # noqa: E402
from prompt_2026 import SYSTEM_PROMPT_2026, construir_entrada_con_prompt  # noqa: E402

import run_sweep_2026  # noqa: E402
import run_control_prompt_original as rcpo  # noqa: E402
import run_control  # noqa: E402
import comparar_control as cc  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Dobles de tokenizer (mismos que tests/test_prompt_2026.py)
# --------------------------------------------------------------------------

class TokenizerConChat:
    """Doble mínimo de un tokenizer instruct: expone chat_template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def __init__(self):
        self.mensajes_recibidos = None
        self.kwargs_recibidos = None

    def apply_chat_template(self, mensajes, **kwargs):
        self.mensajes_recibidos = mensajes
        self.kwargs_recibidos = kwargs
        return {"input_ids": [[1, 2, 3]]}

    def __call__(self, texto, **kwargs):
        raise AssertionError("no se debe tokenizar en crudo un modelo con chat_template")


class TokenizerBase:
    """Doble mínimo de un modelo base: chat_template es None."""

    chat_template = None

    def __init__(self):
        self.texto_recibido = None

    def apply_chat_template(self, mensajes, **kwargs):
        raise AssertionError("no se debe usar chat_template en un modelo base")

    def __call__(self, texto, **kwargs):
        self.texto_recibido = texto
        return {"input_ids": [[4, 5]]}


# --------------------------------------------------------------------------
# Infraestructura factorizada (habilita la corrida de control sin copiarla)
# --------------------------------------------------------------------------

def test_construir_entrada_2026_sigue_usando_el_prompt_2026_por_defecto():
    """La factorización no puede cambiar el comportamiento del barrido principal."""
    from prompt_2026 import construir_entrada
    tok = TokenizerConChat()
    construir_entrada(tok, "Prendé la luz.")
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_2026


def test_construir_entrada_con_prompt_permite_sustituir_el_sistema():
    tok = TokenizerConChat()
    entrada, modo = construir_entrada_con_prompt(tok, "Prendé la luz.", SYSTEM_PROMPT_PAPER)
    assert modo == "chat_template"
    assert tok.mensajes_recibidos[0]["role"] == "system"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_PAPER
    assert tok.mensajes_recibidos[0]["content"] != SYSTEM_PROMPT_2026
    assert tok.mensajes_recibidos[1]["content"] == 'Comando: "Prendé la luz."'


def test_construir_entrada_con_prompt_modo_raw_usa_el_prompt_dado():
    tok = TokenizerBase()
    entrada, modo = construir_entrada_con_prompt(tok, "Apagá la tele.", SYSTEM_PROMPT_PAPER)
    assert modo == "raw_completion"
    assert tok.texto_recibido.startswith(SYSTEM_PROMPT_PAPER)


def test_construir_entrada_con_prompt_rechaza_comando_vacio():
    with pytest.raises(ValueError, match="comando vacío"):
        construir_entrada_con_prompt(TokenizerConChat(), "   ", SYSTEM_PROMPT_PAPER)


def test_evaluar_modelo_acepta_un_constructor_de_entrada_alternativo(monkeypatch):
    """`run_sweep_2026.evaluar_modelo` debe poder recibir un builder de entrada
    distinto sin tocar nada más del pipeline (mismo dataset, mismo scoring)."""
    llamadas = []

    class FakeCtx:
        modelo = por_nombre("SmolLM2-360M-Instruct")
        tokenizer = object()
        transformers_version = "4.57.6"

    def fake_abrir_contexto(_modelo):
        return FakeCtx()

    def fake_generar_respuesta(_ctx, _entrada):
        return (
            '{"intent": "encender", "dispositivo": "luz", "ubicacion": "living", '
            '"valor": null, "unidad": null}',
            0.01,
        )

    def constructor_de_prueba(_tokenizer, comando):
        llamadas.append(comando)
        return {"input_ids": [[1]]}, "chat_template"

    monkeypatch.setattr(run_sweep_2026, "_abrir_contexto", fake_abrir_contexto)
    monkeypatch.setattr(run_sweep_2026, "_generar_respuesta", fake_generar_respuesta)

    dataset = run_sweep_2026.cargar_dataset()
    filas = run_sweep_2026.evaluar_modelo(
        por_nombre("SmolLM2-360M-Instruct"), dataset, construir_entrada=constructor_de_prueba
    )
    assert len(llamadas) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert len(filas) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert all(f["modo_prompting"] == "chat_template" for f in filas)


# --------------------------------------------------------------------------
# Task 1: src/run_control_prompt_original.py
# --------------------------------------------------------------------------

def test_anclas_control_son_exactamente_las_tres_de_continuidad():
    assert set(rcpo.ANCLAS_CONTROL) == {
        "SmolLM2-360M-Instruct", "SmolLM2-1.7B-Instruct", "Qwen2.5-1.5B-Instruct",
    }
    assert len(rcpo.ANCLAS_CONTROL) == 3


def test_ruta_control_usa_el_slug_bajo_control_prompt_original():
    ruta = rcpo.ruta_control("Qwen2.5-1.5B-Instruct")
    assert ruta.name == f"{slug('Qwen2.5-1.5B-Instruct')}.csv"
    assert ruta.parent.name == "control_prompt_original"
    assert ruta.parent.parent.name == "2026"


def test_verificar_es_ancla_rechaza_un_modelo_fuera_de_los_tres():
    otro = por_nombre("LFM2.5-230M")
    with pytest.raises(ValueError, match="no es uno de los tres anclajes"):
        rcpo.verificar_es_ancla(otro)


def test_verificar_es_ancla_acepta_los_tres_anclajes():
    for nombre in rcpo.ANCLAS_CONTROL:
        rcpo.verificar_es_ancla(por_nombre(nombre))  # no debe lanzar


def test_construir_entrada_original_usa_el_prompt_del_paper_no_el_2026():
    tok = TokenizerConChat()
    entrada, modo = rcpo.construir_entrada_original(tok, "Prendé la luz.")
    assert modo == "chat_template"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_PAPER
    assert tok.mensajes_recibidos[0]["content"] != SYSTEM_PROMPT_2026


def test_main_escribe_el_csv_de_control_con_las_columnas_congeladas(monkeypatch, tmp_path):
    """CLI de punta a punta sin inferencia real: `evaluar_modelo` se reemplaza
    por un fake que arma filas con `armar_fila` real (mismo scoring que el
    barrido principal), y la escritura va a `tmp_path`, no a `data/2026/`."""
    ruta_tmp = tmp_path / "smollm2-360m-instruct.csv"
    monkeypatch.setattr(rcpo, "ruta_control", lambda nombre: ruta_tmp)

    dataset = run_sweep_2026.cargar_dataset()

    def fake_evaluar_modelo(modelo_, dataset_, construir_entrada=None):
        assert construir_entrada is rcpo.construir_entrada_original
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

    monkeypatch.setattr(rcpo, "evaluar_modelo", fake_evaluar_modelo)
    monkeypatch.setattr(rcpo, "cargar_dataset", lambda: dataset)
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "SmolLM2-360M-Instruct",
    ])

    codigo = rcpo.main()
    assert codigo == 0
    df = pd.read_csv(ruta_tmp)
    assert list(df.columns) == run_sweep_2026.COLUMNAS_DETALLE
    assert len(df) == run_sweep_2026.N_COMANDOS_ESPERADO
    assert sorted(df["idx"]) == list(range(run_sweep_2026.N_COMANDOS_ESPERADO))


def test_main_rechaza_un_modelo_fuera_de_los_tres_anclajes(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "LFM2.5-230M",
    ])
    with pytest.raises(ValueError, match="no es uno de los tres anclajes"):
        rcpo.main()


def test_main_saltea_si_el_csv_de_control_ya_esta_completo(monkeypatch, tmp_path):
    ruta_tmp = tmp_path / "ya-completo.csv"
    pd.DataFrame(
        {c: [None] * run_sweep_2026.N_COMANDOS_ESPERADO for c in run_sweep_2026.COLUMNAS_DETALLE}
        | {"idx": list(range(run_sweep_2026.N_COMANDOS_ESPERADO))}
    ).to_csv(ruta_tmp, index=False)

    monkeypatch.setattr(rcpo, "ruta_control", lambda nombre: ruta_tmp)

    def fallar_si_se_llama(*_a, **_k):
        raise AssertionError("no debería evaluarse: el CSV de control ya está completo")

    monkeypatch.setattr(rcpo, "evaluar_modelo", fallar_si_se_llama)
    monkeypatch.setattr(sys, "argv", [
        "run_control_prompt_original.py", "--modelo", "SmolLM2-360M-Instruct",
    ])
    assert rcpo.main() == 0


def test_el_modulo_de_control_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "run_control_prompt_original.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea


# --------------------------------------------------------------------------
# Task 2: docker/run_control.py
# --------------------------------------------------------------------------

def test_modelos_control_son_los_tres_anclajes_en_orden_fijo():
    nombres = [m.nombre for m in run_control.modelos_control()]
    assert nombres == list(rcpo.ANCLAS_CONTROL)


def test_comando_run_control_respeta_el_mismo_envolvente_de_recursos():
    modelo = por_nombre("SmolLM2-360M-Instruct")
    cmd = run_control.comando_run_control(modelo, RAIZ)
    assert "--memory=8g" in cmd and "--cpus=2" in cmd
    assert f"--cpuset-cpus={run_control.CPUSET_POR_DEFECTO}" in cmd
    assert run_control.CPUSET_POR_DEFECTO == "0-1"
    assert "--rm" in cmd


def test_comando_run_control_monta_datos_y_cache_compartida():
    cmd = run_control.comando_run_control(por_nombre("SmolLM2-360M-Instruct"), RAIZ)
    montajes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert any(m.endswith(":/app/data") for m in montajes)
    assert any(m.endswith(":/app/.hf_cache") for m in montajes)


def test_comando_run_control_monta_src_de_solo_lectura_con_el_envolvente_completo():
    """Las imágenes `slm-domotica-2026:*` hornean `src/` al build time y por lo
    tanto no tienen `run_control_prompt_original.py` (postdata esas imágenes).
    En vez de reconstruir, se monta el `src/` del working tree de solo
    lectura, para que el contenedor ejecute exactamente el código versionado
    (incluidas las funciones aditivas de `prompt_2026`/`run_sweep_2026` de las
    que depende el entrypoint de control)."""
    cmd = run_control.comando_run_control(por_nombre("SmolLM2-360M-Instruct"), RAIZ)
    montajes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert any(m.endswith(":/app/src:ro") for m in montajes)
    assert "--cpuset-cpus=0-1" in cmd
    assert "--memory=8g" in cmd
    assert "--cpus=2" in cmd


def test_comando_run_control_no_usa_env_file_ninguno_de_los_tres_es_gated():
    for nombre in rcpo.ANCLAS_CONTROL:
        cmd = run_control.comando_run_control(por_nombre(nombre), RAIZ)
        assert "--env-file" not in cmd


def test_comando_run_control_invoca_el_entrypoint_de_control_no_el_del_barrido():
    modelo = por_nombre("SmolLM2-360M-Instruct")
    cmd = run_control.comando_run_control(modelo, RAIZ)
    assert cmd[-3:] == ["src/run_control_prompt_original.py", "--modelo", modelo.nombre]


def test_comando_run_control_reusa_la_imagen_ya_construida_sin_rebuild():
    from build_all import tag_imagen
    modelo = por_nombre("Qwen2.5-1.5B-Instruct")
    cmd = run_control.comando_run_control(modelo, RAIZ)
    assert tag_imagen(modelo) in cmd
    assert tag_imagen(modelo) == "slm-domotica-2026:qwen2-5-1-5b-instruct"
    assert cmd[:2] == ["docker", "run"], "la corrida de control nunca hace build"


def test_dry_run_imprime_tres_comandos_con_el_mismo_envolvente_de_recursos(capsys):
    codigo = run_control.ejecutar_control(run_control.modelos_control(), force=False, dry_run=True)
    assert codigo == 0
    lineas_comando = [
        linea for linea in capsys.readouterr().out.splitlines() if linea.startswith("docker run")
    ]
    assert len(lineas_comando) == 3
    for linea in lineas_comando:
        assert "--memory=8g" in linea and "--cpus=2" in linea
        assert "--cpuset-cpus=0-1" in linea


def test_ejecutar_control_no_corta_al_primer_fallo_y_registra_verbatim(monkeypatch, tmp_path):
    corridos = []
    objetivo = run_control.modelos_control()[1].nombre

    def falso_correr_modelo(modelo, _posicion, _force, _dry_run, _cpuset):
        corridos.append(modelo.nombre)
        return (1, "boom textual") if modelo.nombre == objetivo else (0, "")

    monkeypatch.setattr(run_control, "_correr_modelo", falso_correr_modelo)
    monkeypatch.setattr(run_control, "_csv_ya_completo", lambda _ruta: False)
    monkeypatch.setattr(run_control, "FALLOS_PATH", tmp_path / "fallos_control.json")

    codigo = run_control.ejecutar_control(run_control.modelos_control(), force=False, dry_run=False)
    assert codigo == 1
    assert corridos == [m.nombre for m in run_control.modelos_control()]

    fallos = json.loads((tmp_path / "fallos_control.json").read_text(encoding="utf-8"))
    assert len(fallos) == 1
    assert fallos[0]["modelo"] == objetivo
    assert fallos[0]["error_textual"] == "boom textual"
    assert set(fallos[0]) == {
        "modelo", "hf_repo_id", "transformers_pin", "codigo_salida", "error_textual", "momento_iso",
    }


def test_ejecutar_control_sin_fallos_escribe_lista_vacia_siempre(monkeypatch, tmp_path):
    monkeypatch.setattr(run_control, "_correr_modelo", lambda *_a, **_k: (0, ""))
    monkeypatch.setattr(run_control, "_csv_ya_completo", lambda _ruta: False)
    monkeypatch.setattr(run_control, "FALLOS_PATH", tmp_path / "fallos_control.json")
    assert run_control.ejecutar_control(run_control.modelos_control(), force=False, dry_run=False) == 0
    assert json.loads((tmp_path / "fallos_control.json").read_text(encoding="utf-8")) == []


def test_dry_run_no_escribe_fallos_control(monkeypatch, tmp_path):
    monkeypatch.setattr(run_control, "FALLOS_PATH", tmp_path / "fallos_control.json")
    assert run_control.ejecutar_control(run_control.modelos_control(), force=False, dry_run=True) == 0
    assert not (tmp_path / "fallos_control.json").exists()


def test_ejecutar_control_saltea_un_anclaje_con_csv_ya_completo(monkeypatch, tmp_path):
    monkeypatch.setattr(run_control, "FALLOS_PATH", tmp_path / "fallos_control.json")
    llamados = []
    modelos = run_control.modelos_control()
    completo = modelos[0].nombre

    def falso_correr_modelo(modelo, *_a, **_k):
        llamados.append(modelo.nombre)
        return 0, ""

    monkeypatch.setattr(run_control, "_correr_modelo", falso_correr_modelo)
    monkeypatch.setattr(run_control, "ruta_control", lambda nombre: tmp_path / f"{slug(nombre)}.csv")
    monkeypatch.setattr(run_control, "_csv_ya_completo",
                        lambda ruta: ruta.name == f"{slug(completo)}.csv")

    codigo = run_control.ejecutar_control(modelos, force=False, dry_run=False)
    assert codigo == 0
    assert completo not in llamados
    assert len(llamados) == len(modelos) - 1


def test_ejecutar_control_con_force_no_saltea_aunque_este_completo(monkeypatch, tmp_path):
    monkeypatch.setattr(run_control, "FALLOS_PATH", tmp_path / "fallos_control.json")
    llamados = []
    monkeypatch.setattr(run_control, "_correr_modelo",
                        lambda modelo, *_a, **_k: (llamados.append(modelo.nombre), (0, ""))[1])
    monkeypatch.setattr(run_control, "_csv_ya_completo", lambda _ruta: True)
    modelos = run_control.modelos_control()
    codigo = run_control.ejecutar_control(modelos, force=True, dry_run=False)
    assert codigo == 0
    assert len(llamados) == len(modelos)


def test_no_hay_credenciales_en_run_control():
    import re
    fuente = (RAIZ / "docker" / "run_control.py").read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", fuente)
    assert "ARG HF_TOKEN" not in fuente and "ENV HF_TOKEN" not in fuente


# --------------------------------------------------------------------------
# Task 3: src/comparar_control.py
# --------------------------------------------------------------------------

def test_cargar_publicado_indexa_por_modelo(tmp_path):
    ruta = tmp_path / "resumen.json"
    ruta.write_text(json.dumps([
        {"modelo": "SmolLM2-360M-Instruct", "exact_match_pct": 18.8,
         "json_valido_pct": 100.0, "avg_latencia_s": 15.617},
    ]), encoding="utf-8")
    publicado = cc.cargar_publicado(ruta)
    assert publicado["SmolLM2-360M-Instruct"]["exact_match_pct"] == 18.8


def test_cargar_publicado_usa_el_f0_real_por_defecto():
    """La fuente única de las cifras publicadas es el F0 real
    (data/resultados_experimento_resumen.json); nunca un número hardcodeado."""
    publicado = cc.cargar_publicado()
    assert publicado["SmolLM2-360M-Instruct"]["exact_match_pct"] == 18.8
    assert publicado["Qwen2.5-1.5B-Instruct"]["exact_match_pct"] == 50.0
    assert publicado["SmolLM2-1.7B-Instruct"]["exact_match_pct"] == 59.4
    for fila in publicado.values():
        assert fila["json_valido_pct"] == 100.0


def _df_control(n_exactos: int, n_json_validos: int, n: int = 4) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_exact": i < n_exactos,
            "json_valido": i < n_json_validos,
            "latencia_s": 1.0 + i,
        }
        for i in range(n)
    ])


def test_metricas_de_detalle_calcula_los_tres_porcentajes():
    m = cc.metricas_de_detalle(_df_control(n_exactos=1, n_json_validos=2, n=4))
    assert m["n"] == 4
    assert m["exact_match_pct"] == 25.0
    assert m["json_valido_pct"] == 50.0
    assert m["avg_latencia_s"] == 2.5


def test_cargar_banda_devuelve_none_si_no_existe(tmp_path):
    assert cc.cargar_banda(tmp_path, "SmolLM2-360M-Instruct") is None


def test_cargar_banda_lee_el_csv_del_slug(tmp_path):
    ruta = tmp_path / f"{slug('SmolLM2-360M-Instruct')}.csv"
    _df_control(2, 3, 4).to_csv(ruta, index=False)
    m = cc.cargar_banda(tmp_path, "SmolLM2-360M-Instruct")
    assert m["n"] == 4


def test_comparar_anclaje_rechaza_un_modelo_sin_cifra_publicada():
    with pytest.raises(ValueError):
        cc.comparar_anclaje("LFM2.5-230M", cc.cargar_publicado())


def test_comparar_anclaje_calcula_los_tres_deltas(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "DIR_CONTROL", tmp_path / "control")
    monkeypatch.setattr(cc, "DIR_DETALLE_NUEVO", tmp_path / "nuevo")
    (tmp_path / "control").mkdir()
    (tmp_path / "nuevo").mkdir()
    slug_m = slug("SmolLM2-360M-Instruct")
    _df_control(n_exactos=8, n_json_validos=32, n=32).to_csv(
        tmp_path / "control" / f"{slug_m}.csv", index=False)
    _df_control(n_exactos=8, n_json_validos=8, n=32).to_csv(
        tmp_path / "nuevo" / f"{slug_m}.csv", index=False)

    publicado = cc.cargar_publicado()
    resultado = cc.comparar_anclaje("SmolLM2-360M-Instruct", publicado)
    assert resultado["control_prompt_original"]["exact_match_pct"] == 25.0
    assert resultado["nuevo_prompt_2026"]["exact_match_pct"] == 25.0
    d = resultado["deltas_exact_match_pp"]
    assert d["control_vs_publicado"] == round(25.0 - 18.8, 1)
    assert d["nuevo_vs_publicado"] == round(25.0 - 18.8, 1)
    assert d["nuevo_vs_control"] == 0.0


def test_comparar_anclaje_con_datos_reales_de_las_tres_bandas():
    """Sin monkeypatch: usa los CSV reales de data/2026/detalle/ y
    data/2026/control_prompt_original/ (ambos commiteados, de solo lectura).

    Actualizado tras el subtask 18: la corrida de control real ya se ejecutó
    (el orquestador la lanzó después del reporte de ese subtask, ver commits
    `544b988`/`9f584a6`/`3fe0ac3`) y NO reprodujo la cifra publicada -- esa
    divergencia es precisamente la evidencia que motiva el Delta 05. La banda
    ya no da `None`; se compara contra el propio CSV en vez de hardcodear un
    número, para no duplicar el dato real en dos lugares."""
    ruta_control_real = RAIZ / "data" / "2026" / "control_prompt_original" / "smollm2-360m-instruct.csv"
    metricas_control = cc.metricas_de_detalle(pd.read_csv(ruta_control_real))

    publicado = cc.cargar_publicado()
    resultado = cc.comparar_anclaje("SmolLM2-360M-Instruct", publicado)
    assert resultado["nuevo_prompt_2026"] is not None
    assert resultado["nuevo_prompt_2026"]["n"] == 32
    assert resultado["control_prompt_original"] == metricas_control
    assert resultado["deltas_exact_match_pp"]["control_vs_publicado"] == round(
        metricas_control["exact_match_pct"] - resultado["publicado"]["exact_match_pct"], 1
    )


def test_escribir_comparacion_escribe_json_indentado(tmp_path):
    comparacion = {"anclajes": [], "nota_latencia": "x"}
    salida = tmp_path / "out.json"
    cc.escribir_comparacion(comparacion, salida)
    assert json.loads(salida.read_text(encoding="utf-8")) == comparacion


def test_imprimir_tabla_no_crashea_con_bandas_faltantes(capsys):
    comparacion = {
        "anclajes": [{
            "modelo": "X", "publicado": {"exact_match_pct": 10.0},
            "control_prompt_original": None, "nuevo_prompt_2026": None,
            "deltas_exact_match_pp": {
                "control_vs_publicado": None, "nuevo_vs_publicado": None, "nuevo_vs_control": None,
            },
        }],
        "nota_latencia": "nota de prueba",
    }
    cc.imprimir_tabla(comparacion)
    assert "nota de prueba" in capsys.readouterr().out


def test_nota_latencia_advierte_sobre_no_comparabilidad():
    texto = cc.NOTA_LATENCIA.lower()
    assert "latencia" in texto
    assert "no" in texto
    assert "ocios" in texto  # ocioso/ociosa


def test_main_escribe_y_no_toca_el_repo(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cc, "SALIDA_PATH", tmp_path / "comparacion.json")
    codigo = cc.main()
    assert codigo == 0
    assert (tmp_path / "comparacion.json").exists()
    assert "modelo" in capsys.readouterr().out.lower()


def test_el_modulo_de_comparacion_se_importa_sin_torch_ni_transformers():
    fuente = (RAIZ / "src" / "comparar_control.py").read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea
