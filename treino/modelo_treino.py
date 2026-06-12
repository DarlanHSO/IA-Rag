import os
import pickle
import tempfile
import warnings

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ACCESS_KEY", ""))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_SECRET_KEY", ""))
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))

import pandas as pd
from minio import Minio

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:3000"))
mlflow.set_experiment("youtube_trending_classification")

client = Minio(
    f"{os.getenv('MINIO_HOST', 'localhost')}:{os.getenv('MINIO_PORT', '9000')}",
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False,
)


def carregar_dados_gold():
    print("Baixando dados da camada Gold...")
    with tempfile.TemporaryDirectory() as temp_dir:
        gold_path = os.path.join(temp_dir, "trending_gold.parquet")
        client.fget_object("gold", "trending_gold.parquet", gold_path)
        df = pd.read_parquet(gold_path)
    print("Dados carregados com sucesso.")
    return df


def preparar_dados(df):
    print("Iniciando pré-processamento...")

    # viral = views > 1M
    df["viral"] = (df["views"] > 1_000_000).astype(int)

    df["title_length"] = df["title"].astype(str).apply(len)

    # tags vêm como "[none]" quando não informadas no dataset
    df["num_tags"] = df["tags"].astype(str).apply(
        lambda x: 0 if x.strip().lower() == "[none]" else len(x.split("|"))
    )

    df["description_length"] = df["description"].astype(str).apply(len)

    safe_views = df["views"].replace(0, 1)
    df["likes_ratio"] = df["likes"] / safe_views
    df["comments_ratio"] = df["comments"] / safe_views

    features = [
        "likes",
        "comments",
        "title_length",
        "num_tags",
        "description_length",
        "likes_ratio",
        "comments_ratio"
    ]

    available_features = [col for col in features if col in df.columns]
    print("Features usadas:", available_features)

    X = df[available_features].fillna(0)
    y = df["viral"]

    print("Pré-processamento concluído.")
    return X, y


def treino_avaliar_modelo(model_name, model, X_train, X_test, y_train, y_test):
    print(f"\nTreinando modelo: {model_name}")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_name", model_name)

        if hasattr(model, "get_params"):
            for param_name, param_value in model.get_params().items():
                mlflow.log_param(param_name, param_value)

        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, zero_division=0))

        return {
            "model_name": model_name,
            "accuracy":   accuracy,
            "precision":  precision,
            "recall":     recall,
            "f1_score":   f1
        }


def main():
    print("Iniciando pipeline de treinamento...")
    df = carregar_dados_gold()
    X, y = preparar_dados(df)

    # 80/20, estratificado para preservar proporção de virais
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTamanho dos conjuntos:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    all_results = []
    for model_name, model in models.items():
        result = treino_avaliar_modelo(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results).sort_values(by="f1_score", ascending=False)
    print("\nResumo final dos modelos:")
    print(results_df)

    print(f"\nMelhor modelo: {results_df.iloc[0]['model_name']}")
    print("\nPipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
