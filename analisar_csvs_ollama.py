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
        # Enviar o pedido POST para a API
        response = requests.post(url, json=payload)
        # Verifica se a resposta é bem-sucedida
        response.raise_for_status()
        
        # Converte a resposta para JSON
        data = response.json()
        
        # Retorna a mensagem gerada pela API
        return data.get("message", {}).get("content", "Resposta não encontrada.")
    
    except requests.exceptions.RequestException as e:
        # Trata qualquer erro durante a requisição
        return f"Erro ao chamar a API: {str(e)}"

def analisar_csv(caminho_csv: str, url: str, model: str):
    try:
        # Carregar o arquivo CSV em um DataFrame
        df = pd.read_csv(caminho_csv)

        # Análise de sentimentos para cada comentário
        resultados = []
        total_comentarios = len(df)
        
        print(f"Iniciando análise de sentimentos em {total_comentarios} comentários...\n")
        
        for idx, comentario in enumerate(df['comment']):  # Alterar para o nome correto da coluna
            user_msg = f"Classify the following YouTube music video comment as Positive, Negative, or Neutral, and justify your answer in a short sentence.\n\n{comentario}"
            resposta_api = chamar_api(url, model, "You are an expert in sentiment analysis.", user_msg)
            
            resultados.append({
                "comentario": comentario,
                "resposta_api": resposta_api
            })
            
            # Exibir o progresso com a quantidade de comentários analisados
            if (idx + 1) % 10 == 0:  # Exibe a cada 10 comentários
                print(f"Analisado {idx + 1}/{total_comentarios} comentários...")

        # Resultado final
        print("\nAnálise concluída!\n")
        
        # Salvar os resultados em um arquivo CSV
        resultados_df = pd.DataFrame(resultados)
        resultados_df.to_csv('analise_resultados.csv', index=False, encoding="utf-8-sig")
        print("Análise salva em 'analise_resultados.csv'")

        return resultados
    
    except Exception as e:
        return f"Erro ao processar o arquivo CSV: {str(e)}"

if __name__ == "__main__":
    # URL da API
    url = "http://127.0.0.1:11434/api/chat"
    
    # Definir os parâmetros para a chamada
    model = "llama3:8b"
    
    # Caminho do arquivo CSV
    caminho_csv = 'the weeknd - dancing in the flames (official music video) comments.csv'  # Substitua pelo caminho do seu arquivo CSV

    # Realizar a análise de sentimentos do CSV
    analise_resultados = analisar_csv(caminho_csv, url, model)
    
    # Exibir os resultados
    for resultado in analise_resultados:
        print(f"Comentário: {resultado['comentario']}")
        print(f"Resposta da API: {resultado['resposta_api']}\n")
