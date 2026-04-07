# Documentação de Governança de Dados - sprint 3

Este documento define as regras de negócio, qualidade e arquitetura aplicadas ao pipeline do projeto ia-rag.

## 1. Arquitetura Medallion e linhagem

o fluxo de dados utiliza o minio como data lake, dividido em três restrições de acesso e mutabilidade:

* **bronze (raw):** camada imutável. armazena os csvs originais do kaggle e o dicionário json da api. centraliza a origem para atuar como fonte única da verdade (ssot), forçando as próximas etapas a consumirem do data lake e não de máquinas locais.
* **silver (trusted):** camada de qualidade. os dados regionais são consolidados. a transição para o formato parquet ocorre aqui para preservar a tipagem columnar construída no pandas e reduzir drasticamente o custo de i/o nas leituras futuras.
* **gold (business):** camada de consumo. o dado é formatado exclusivamente para a indexação no banco de vetores do rag.

## 2. Regras de qualidade (data quality)

para um dado ser promovido da camada bronze até a gold, ele passa obrigatoriamente pelos seguintes contratos:

* **remoção de viés:** a coluna de dislikes é expurgada do pipeline devido à sua descontinuação pública pelo youtube, evitando viés analítico e dados incompletos.
* **filtragem de ruído:** registros sem visualizações (views <= 0) e linhas duplicadas são descartados por serem anomalias estatísticas que não contribuem para o modelo.
* **completude semântica:** campos vazios ou corrompidos (como "[none]") em tags e descrições são padronizados para a string "não informado", mitigando falhas de contexto na ia.
* **tipagem estrita:** datas são convertidas forçosamente para datetime e métricas para inteiros, prevenindo quebras de esquema nas camadas de consumo.
* **enriquecimento obrigatório:** o id numérico da categoria é mapeado para seu nome em texto usando o dicionário oficial, agregando o valor semântico necessário para a ia entender o assunto.

## 3. Regras de negócio para inteligência artificial (gold)

* **descarte estrutural:** chaves numéricas isoladas, como channel_id, são removidas por não agregarem valor na busca semântica.
* **densidade de contexto:** os metadados úteis (título, categoria, canal, tags e descrição) são consolidados em uma única string textual (texto_rag) para otimizar a criação de embeddings pelo modelo de linguagem.

## 4. Segurança e infraestrutura

* **processamento stateless:** todas as operações de i/o de arquivos brutos ocorrem em memória/diretórios temporários, prevenindo o esgotamento do disco local com resíduos de processamento (como zips e csvs descartados).
* **proteção de credenciais:** o acesso ao kaggle e ao minio não possui hardcode no repositório. as chaves são injetadas exclusivamente via variáveis de ambiente (.env).
* **resiliência de caminhos:** a leitura de dados resolve os diretórios dinamicamente. isso garante que o pipeline não falhe por erros de rota independente de onde o terminal seja executado.
