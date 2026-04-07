import os
import json
from pathlib import Path
from googleapiclient.discovery import build
from dotenv import load_dotenv

# garante que o script encontre as credenciais independente do diretório de execução
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_master_categories():
    youtube = build("youtube", "v3", developerKey=API_KEY)
    master_dict = {}
    
    # a api do youtube retorna as mesmas chaves globais independentemente da região consultada
    region = "US"
    
    try:
        request = youtube.videoCategories().list(part="snippet", regionCode=region)
        response = request.execute()
        for item in response.get("items", []):
            cat_id = int(item["id"])
            cat_name = item["snippet"]["title"]
            master_dict[cat_id] = cat_name
    except Exception as e:
        print(f"erro ao acessar a api: {e}")
            
    return master_dict

if __name__ == "__main__":
    categorias = get_master_categories()
    
    # fixa o destino na raiz do projeto para garantir que o script de ingestão bronze localize o arquivo depois
    output_dir = BASE_DIR / "data" / "bronze"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "categories_official.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(categorias, f, indent=4, ensure_ascii=False)
        
    print(f"dicionário gerado com sucesso.")