Os modelos de machine learning treinados na plataforma são: LogisticRegression, DecisionTreeClassifier e RandomForestClassifier, todos do scikit-learn. O problema é classificação binária: prever se um vídeo é viral (views acima de 1.000.000).

As features utilizadas no treinamento são: likes, comments, title_length, num_tags, description_length, likes_ratio e comments_ratio.

As métricas registradas no MLflow para cada modelo são: accuracy (proporção de acertos), precision (virais previstos corretamente), recall (virais reais identificados) e f1_score (média harmônica entre precision e recall). Os experimentos ficam no MLflow sob o nome youtube_trending_classification.
