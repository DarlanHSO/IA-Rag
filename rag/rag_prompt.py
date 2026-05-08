import ollama

from pymilvus import (
    connections,
    Collection
)

# =========================================================
# CONFIG
# =========================================================

EMBED_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3"

MILVUS_HOST = "localhost"

MILVUS_PORT = "19530"

COLLECTION_NAME = "youtube_trending"

TOP_K = 5

# =========================================================
# INTRO
# =========================================================

print("\n===================================")
print("YOUTUBE VIRAL RAG")
print("===================================\n")

print("Sistema iniciado com sucesso!")
print()

print("Exemplos:")
print("- Me dê ideias de vídeos virais de Minecraft")
print("- Sugira títulos para vídeos de IA")
print("- Temas em alta para shorts")
print("- Títulos virais para GTA RP")
print("- Ideias para vídeos dark")
print()

# =========================================================
# CONECTAR MILVUS
# =========================================================

print("\n===================================")
print("CONECTANDO AO MILVUS")
print("===================================\n")

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

print("Conectado ao Milvus!")

collection = Collection(COLLECTION_NAME)

collection.load()

print("Collection carregada!")

# =========================================================
# SEARCH PARAMS
# =========================================================

search_params = {

    "metric_type": "COSINE",

    "params": {
        "ef": 128
    }
}

# =========================================================
# LOOP PRINCIPAL
# =========================================================

while True:

    print("\n===================================")

    user_query = input("Pergunta: ")

    # =====================================================
    # SAIR
    # =====================================================

    if user_query.lower() in [

        "exit",

        "quit",

        "sair"
    ]:

        print("\nEncerrando sistema...")
        break

    # =====================================================
    # MOSTRAR QUERY
    # =====================================================

    print("\n===================================")
    print("QUERY ORIGINAL")
    print("===================================\n")

    print(user_query)

    # =====================================================
    # EXPANDIR QUERY
    # =====================================================

    rag_query = f"""
    Usuário busca ideias de conteúdo viral para YouTube.

    Pedido:
    {user_query}

    Buscar:
    - vídeos similares
    - títulos engajantes
    - temas virais
    - conteúdo com alta retenção
    - tendências relacionadas
    - formatos que aumentam CTR
    - vídeos com potencial viral
    """

    print("\n===================================")
    print("QUERY EXPANDIDA")
    print("===================================\n")

    print(rag_query)

    # =====================================================
    # GERAR EMBEDDING
    # =====================================================

    print("\n===================================")
    print("GERANDO EMBEDDING")
    print("===================================\n")

    try:

        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=rag_query
        )

        embedding = response["embedding"]

        print("Embedding gerado!")

        print(
            f"Dimensão embedding: "
            f"{len(embedding)}"
        )

    except Exception as e:

        print("\n===================================")
        print("ERRO AO GERAR EMBEDDING")
        print("===================================\n")

        print(e)

        continue

    # =====================================================
    # BUSCA NO MILVUS
    # =====================================================

    print("\n===================================")
    print("BUSCANDO NO MILVUS")
    print("===================================\n")

    try:

        results = collection.search(

            data=[embedding],

            anns_field="embedding",

            param=search_params,

            limit=TOP_K,

            output_fields=[

                "video_id",

                "title",

                "channel_title",

                "category_name",

                "country",

                "texto_rag"
            ]
        )

    except Exception as e:

        print("\n===================================")
        print("ERRO NO MILVUS")
        print("===================================\n")

        print(e)

        continue

    # =====================================================
    # PROCESSAR RESULTADOS
    # =====================================================

    print("\n===================================")
    print("RESULTADOS ENCONTRADOS")
    print("===================================\n")

    rag_context = ""

    retrieved_documents = []

    for hits in results:

        for idx, hit in enumerate(hits):

            entity = hit.entity

            video_data = {

                "video_id":
                    entity.get("video_id"),

                "title":
                    entity.get("title"),

                "channel_title":
                    entity.get("channel_title"),

                "category_name":
                    entity.get("category_name"),

                "country":
                    entity.get("country"),

                "texto_rag":
                    entity.get("texto_rag"),

                "score":
                    hit.distance
            }

            retrieved_documents.append(
                video_data
            )

            # =================================================
            # PRINT RESULTADO
            # =================================================

            print(f"RESULTADO #{idx + 1}")

            print(
                f"Similaridade: "
                f"{video_data['score']}"
            )

            print(
                f"Título: "
                f"{video_data['title']}"
            )

            print(
                f"Canal: "
                f"{video_data['channel_title']}"
            )

            print(
                f"Categoria: "
                f"{video_data['category_name']}"
            )

            print(
                f"País: "
                f"{video_data['country']}"
            )

            print(
                "\n-----------------------------------\n"
            )

            # =================================================
            # MONTAR CONTEXTO
            # =================================================

            rag_context += f"""

            VÍDEO #{idx + 1}

            Similaridade:
            {video_data['score']}

            Título:
            {video_data['title']}

            Canal:
            {video_data['channel_title']}

            Categoria:
            {video_data['category_name']}

            País:
            {video_data['country']}

            Conteúdo:
            {video_data['texto_rag']}

            ============================================
            """

    # =====================================================
    # MOSTRAR CONTEXTO
    # =====================================================

    print("\n===================================")
    print("CONTEXTO RAG")
    print("===================================\n")

    print(rag_context[:3000])

    # =====================================================
    # PROMPT FINAL
    # =====================================================

    print("\n===================================")
    print("MONTANDO PROMPT")
    print("===================================\n")

    rag_prompt = f"""
    Você é um especialista em YouTube viral,
    SEO para YouTube, retenção de audiência,
    CTR e crescimento de canais.

    O usuário pediu:

    {user_query}

    Você recebeu vídeos reais similares
    encontrados semanticamente no banco vetorial.

    Use os padrões encontrados
    para gerar respostas altamente úteis.

    ================= CONTEXTO =================

    {rag_context}

    ==================================================

    REGRAS:

    - Responda em português brasileiro
    - Seja criativo
    - Seja Breve
    - Analise padrões dos vídeos
    - Foque em retenção e CTR
    - NÃO copie exatamente os títulos
    - Use listas
    - Responda títulos em português brasileiro
    - Sugira o que o usuário pedir entre título, thumbnails, hooks, formatos virais, ideias originais, tags
    - Se você não achar uma resposta, fale que não consigo dar uma boa sugestão

    Se possível forneça:
    - uma pergunta para o usuário perguntando se ele deseja uma sugestão de thumbnail, hooks, tags
    """

    print("Prompt final montado!")

    # =====================================================
    # OLLAMA GENERATION
    # =====================================================

    print("\n===================================")
    print("GERANDO RESPOSTA")
    print("===================================\n")

    try:

        stream = ollama.chat(

            model=LLM_MODEL,

            stream=True,

            messages=[

                {
                    "role": "system",

                    "content": (
                        "Você é um especialista "
                        "em crescimento viral "
                        "no YouTube."
                    )
                },

                {
                    "role": "user",

                    "content": rag_prompt
                }
            ]
        )

        print("\n===================================")
        print("RESPOSTA DO RAG")
        print("===================================\n")

        for chunk in stream:

            content = chunk["message"]["content"]

            print(
                content,
                end="",
                flush=True
            )

        print("\n")

    except Exception as e:

        print("\n===================================")
        print("ERRO NO OLLAMA")
        print("===================================\n")

        print(e)