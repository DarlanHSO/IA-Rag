import sys
import requests

API_URL = "http://localhost:8000"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except AssertionError as e:
        print(f"  {FAIL}  {name}  ->  {e}")
        return False
    except Exception as e:
        print(f"  {FAIL}  {name}  ->  {type(e).__name__}: {e}")
        return False


# ── Health checks ─────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{API_URL}/health", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"
    assert r.json()["status"] == "ok", "campo 'status' != 'ok'"


def test_models():
    r = requests.get(f"{API_URL}/models", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"
    models = r.json().get("available", [])
    assert len(models) > 0, "nenhum modelo disponível no Ollama"
    print(f"       modelos: {models}", end="")


def test_metadata():
    r = requests.get(f"{API_URL}/metadata", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"
    d = r.json()
    assert d["vector_store"]["collection"] == "youtube_trending", "collection errada"
    buckets = d["data_lake"]["buckets"]
    for b in ("bronze", "silver", "gold"):
        assert b in buckets, f"bucket '{b}' não encontrado no MinIO"
    print(f"       buckets: {buckets}", end="")


def test_mlflow_experiments():
    r = requests.get(f"{API_URL}/mlflow/experiments", timeout=30)
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    total_runs = sum(len(e["runs"]) for e in data)
    assert total_runs > 0, "nenhum run registrado no MLflow"
    print(f"       {len(data)} experimentos, {total_runs} runs", end="")


def test_mlflow_metrics():
    r = requests.get(
        f"{API_URL}/mlflow/metrics",
        params={"experiment_name": "youtube_trending_classification"},
        timeout=10,
    )
    assert r.status_code == 200, f"status {r.status_code}"
    runs = r.json().get("runs", [])
    assert len(runs) > 0, "nenhum run no experimento de classificação"
    best = max(runs, key=lambda x: x["metrics"].get("f1_score", 0))
    print(f"       melhor modelo: {best['run_name']} (f1={best['metrics'].get('f1_score', 0):.4f})", end="")


def test_top_videos():
    r = requests.get(f"{API_URL}/videos/top", params={"limit": 5}, timeout=60)
    assert r.status_code == 200, f"status {r.status_code}"
    videos = r.json().get("videos", [])
    assert len(videos) > 0, "nenhum vídeo retornado"
    print(f"       {len(videos)} vídeos", end="")


def test_gradio():
    r = requests.get("http://localhost:7860", timeout=10)
    assert r.status_code == 200, f"status {r.status_code}"


# ── Integração RAG ────────────────────────────────────────────────────────────

def test_query():
    r = requests.post(
        f"{API_URL}/query",
        json={"question": "vídeos virais de gaming no Brasil", "top_k": 3},
        timeout=240,
    )
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "answer" in data, "campo 'answer' ausente"
    assert len(data.get("retrieved_documents", [])) > 0, "nenhum documento recuperado"
    print(f"       {data['latency']['total_ms']:.0f}ms", end="")


def test_query_mlflow():
    r = requests.post(
        f"{API_URL}/query/mlflow",
        json={"question": "qual foi o melhor modelo treinado?"},
        timeout=240,
    )
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "answer" in data, "campo 'answer' ausente"
    assert data["context_runs"] > 0, "nenhum run no contexto"
    print(f"       {data['context_runs']} runs, {data['llm_ms']:.0f}ms", end="")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("GET /health",                               test_health),
    ("GET /models",                               test_models),
    ("GET /metadata  (MinIO buckets + Milvus)",   test_metadata),
    ("GET /mlflow/experiments",                   test_mlflow_experiments),
    ("GET /mlflow/metrics  (classificação)",      test_mlflow_metrics),
    ("GET /videos/top",                           test_top_videos),
    ("Gradio Interface  :7860",                   test_gradio),
    ("POST /query  (RAG YouTube)",                test_query),
    ("POST /query/mlflow  (RAG experimentos)",    test_query_mlflow),
]

if __name__ == "__main__":
    print(f"\nTestando API em {API_URL}\n")
    passed = sum(check(name, fn) for name, fn in TESTS)
    failed = len(TESTS) - passed
    print(f"\n{'-'*50}")
    if failed == 0:
        print(f"\033[92m  Todos os {passed} testes passaram.\033[0m\n")
    else:
        print(f"\033[91m  {failed}/{len(TESTS)} testes falharam.\033[0m\n")
    sys.exit(failed)
