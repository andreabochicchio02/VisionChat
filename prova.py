import requests
import json
import sys

def query_ollama_streaming(prompt, model="llama3.2:3b"):

    # Endpoint Ollama per la generazione di testo
    url = "http://192.168.110.144:11434/api/generate"
    
    # Configurazione della richiesta
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True 
    }
    
    print("\nRisposta:")
    
    try:
        # Esegui la richiesta in streaming
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()  # Verifica eventuali errori HTTP
            
            # Processa la risposta in streaming riga per riga
            for line in response.iter_lines():
                if line:
                    # Decodifica la risposta JSON
                    chunk = json.loads(line)
                    
                    # Stampa il testo in streaming
                    if 'response' in chunk:
                        sys.stdout.write(chunk['response'])
                        sys.stdout.flush()
            print("\n")
    
    except requests.exceptions.ConnectionError:
        print("Errore: impossibile connettersi al server Ollama.")
        print("Assicurati che Ollama sia in esecuzione con il comando 'ollama serve'.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    # Richiedi l'input dell'utente
    prompt = input("Inserisci la tua domanda: ")
    
    # Esegui la query
    query_ollama_streaming(prompt)