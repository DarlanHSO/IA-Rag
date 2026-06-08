import os
import pandas as pd
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", timeout=15, **kwargs)


def _post(path: str, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=180, **kwargs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def fetch_models() -> list[str]:
    try:
        data = _get("/models").json()
        return data.get("available", ["phi3:mini"])
    except Exception:
        return ["phi3:mini"]


# ------------------------------------------------------------------
# Tab 1 — RAG Query
# ------------------------------------------------------------------

def query_rag(question: str, model: str, top_k: int):
    if not question.strip():
        return "Digite uma pergunta.", pd.DataFrame(), ""

    payload = {"question": question, "top_k": int(top_k)}
    if model:
        payload["model"] = model

    try:
        resp = _post("/query", json=payload)
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            return f"Erro {resp.status_code}: {detail}", pd.DataFrame(), ""

        data = resp.json()
        answer = data["answer"]

        df = pd.DataFrame([
            {
                "rank":      d["rank"],
                "score":     d["score"],
                "título":    d["title"],
                "canal":     d["channel"],
                "categoria": d["category"],
                "país":      d["country"],
            }
            for d in data["retrieved_documents"]
        ])

        lat = data["latency"]
        latency = (
            f"Embedding: {lat['embedding_ms']:.0f} ms  |  "
            f"Search: {lat['search_ms']:.0f} ms  |  "
            f"LLM: {lat['llm_ms']:.0f} ms  |  "
            f"Total: {lat['total_ms']:.0f} ms  "
            f"(modelo: {data['models']['llm']})"
        )

        return answer, df, latency

    except Exception as exc:
        return f"Erro de conexão: {exc}", pd.DataFrame(), ""


def refresh_models():
    models = fetch_models()
    return gr.Dropdown(choices=models, value=models[0] if models else None)


# ------------------------------------------------------------------
# Tab 2 — MLflow Experiments
# ------------------------------------------------------------------

def load_experiments():
    try:
        data = _get("/mlflow/experiments").json()
        rows = []
        for exp in data:
            for run in exp["runs"]:
                row = {
                    "experimento": exp["name"],
                    "run":         run["run_name"],
                    "status":      run["status"],
                    "início":      run["start_time"][:19].replace("T", " "),
                }
                for k, v in run["metrics"].items():
                    row[f"metric:{k}"] = round(v, 4)
                for k, v in run["params"].items():
                    row[f"param:{k}"] = v
                rows.append(row)

        if not rows:
            return pd.DataFrame({"info": ["Nenhum experimento registrado no MLflow."]})
        return pd.DataFrame(rows)

    except Exception as exc:
        return pd.DataFrame({"erro": [str(exc)]})


def query_mlflow(question: str, model: str):
    if not question.strip():
        return "Digite uma pergunta sobre os experimentos."
    payload = {"question": question}
    if model:
        payload["model"] = model
    try:
        resp = _post("/query/mlflow", json=payload)
        if resp.status_code != 200:
            return f"Erro {resp.status_code}: {resp.json().get('detail', resp.text)}"
        data = resp.json()
        return f"{data['answer']}\n\n---\n_Modelo: {data['model']} | Runs analisados: {data['context_runs']} | LLM: {data['llm_ms']:.0f}ms_"
    except Exception as exc:
        return f"Erro de conexão: {exc}"


# ------------------------------------------------------------------
# Tab 3 — Top Vídeos
# ------------------------------------------------------------------

def load_top_videos(limit: int):
    try:
        data = _get(f"/videos/top?limit={int(limit)}").json()
        return pd.DataFrame([
            {
                "rank":      v["rank"],
                "views":     f"{v['views']:,}",
                "título":    v["title"],
                "canal":     v["channel"],
                "categoria": v["category"],
                "país":      v["country"],
            }
            for v in data["videos"]
        ])
    except Exception as exc:
        return pd.DataFrame({"erro": [str(exc)]})


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

_initial_models = fetch_models()

with gr.Blocks(title="YouTube Viral RAG") as app:

    gr.Markdown(
        "# YouTube Viral RAG\n"
        "Plataforma de análise de conteúdo viral no YouTube — UniFacens IA"
    )

    # ---- Tab 1 -------------------------------------------------------
    with gr.Tab("RAG Query"):
        with gr.Row():
            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="Pergunta",
                    placeholder="Que tipo de vídeo está viral no Brasil?",
                    lines=3,
                )
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    label="Modelo LLM",
                    choices=_initial_models,
                    value=_initial_models[0] if _initial_models else None,
                )
                top_k_slider = gr.Slider(
                    label="Top-K documentos",
                    minimum=1, maximum=20, step=1, value=5,
                )
                btn_refresh = gr.Button("Atualizar modelos", size="sm")
                btn_query   = gr.Button("Perguntar", variant="primary")

        answer_box  = gr.Textbox(label="Resposta", lines=10, interactive=False)
        latency_box = gr.Textbox(label="Latência", interactive=False, max_lines=1)
        docs_table  = gr.Dataframe(label="Documentos recuperados", interactive=False)

        btn_refresh.click(fn=refresh_models, outputs=model_dd)
        btn_query.click(
            fn=query_rag,
            inputs=[question_box, model_dd, top_k_slider],
            outputs=[answer_box, docs_table, latency_box],
        )

    # ---- Tab 2 -------------------------------------------------------
    with gr.Tab("Perguntar sobre Experimentos"):
        gr.Markdown("Pergunte sobre os experimentos de ML registrados no MLflow (Sprint 4).")
        with gr.Row():
            with gr.Column(scale=3):
                mlflow_question = gr.Textbox(
                    label="Pergunta",
                    placeholder="Qual foi o melhor modelo? Qual o F1-score do RandomForest?",
                    lines=2,
                )
            with gr.Column(scale=1):
                mlflow_model_dd = gr.Dropdown(
                    label="Modelo LLM",
                    choices=_initial_models,
                    value=_initial_models[0] if _initial_models else None,
                )
                btn_mlflow_query = gr.Button("Perguntar", variant="primary")
        mlflow_answer = gr.Textbox(label="Resposta", lines=10, interactive=False)
        btn_mlflow_query.click(
            fn=query_mlflow,
            inputs=[mlflow_question, mlflow_model_dd],
            outputs=mlflow_answer,
        )

    # ---- Tab 3 -------------------------------------------------------
    with gr.Tab("MLflow — Tabela de Runs"):
        gr.Markdown("Visualize todos os experimentos e runs registrados.")
        btn_mlflow   = gr.Button("Carregar experimentos")
        mlflow_table = gr.Dataframe(label="Runs", interactive=False)
        btn_mlflow.click(fn=load_experiments, outputs=mlflow_table)

    # ---- Tab 4 -------------------------------------------------------
    with gr.Tab("Top Vídeos"):
        gr.Markdown("Top vídeos por views — camada Gold do Data Lake.")
        limit_slider = gr.Slider(
            label="Quantidade de vídeos",
            minimum=5, maximum=100, step=5, value=10,
        )
        btn_videos   = gr.Button("Buscar")
        videos_table = gr.Dataframe(label="Top Vídeos", interactive=False)
        btn_videos.click(fn=load_top_videos, inputs=limit_slider, outputs=videos_table)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
