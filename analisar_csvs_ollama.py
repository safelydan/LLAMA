import requests
import json
import pandas as pd
import time

def chamar_api(url: str, model: str, system_msg: str, user_msg: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "resposta não encontrada.")
    
    except requests.exceptions.RequestException as e:
        return f"erro ao chamar a api: {str(e)}"

def analisar_csv(caminho_csv: str, url: str, model: str):
    try:
        df = pd.read_csv(caminho_csv)
        resultados = []
        total_comentarios = len(df)
        
        for idx, comentario in enumerate(df['comment']):
            user_msg = f"classify the following youtube music video comment as positive, negative, or neutral, and justify your answer in a short sentence.\n\n{comentario}"
            resposta_api = chamar_api(url, model, "you are an expert in sentiment analysis.", user_msg)
            
            resultados.append({
                "comentario": comentario,
                "resposta_api": resposta_api
            })
            
            if (idx + 1) % 10 == 0:
                print(f"analisado {idx + 1}/{total_comentarios} comentários...")

        print("\nanálise concluída!\n")
        
        resultados_df = pd.DataFrame(resultados)
        resultados_df.to_csv('analise_resultados.csv', index=False, encoding="utf-8-sig")
        print("análise salva em 'analise_resultados.csv'")

        return resultados
    
    except Exception as e:
        return f"erro ao processar o arquivo csv: {str(e)}"

if __name__ == "__main__":
    url = "http://127.0.0.1:11434/api/chat"
    model = "llama3:8b"
    caminho_csv = 'the weeknd - dancing in the flames (official music video) comments.csv'

    analise_resultados = analisar_csv(caminho_csv, url, model)
    
    for resultado in analise_resultados:
        print(f"comentário: {resultado['comentario']}")
        print(f"resposta da api: {resultado['resposta_api']}\n")
