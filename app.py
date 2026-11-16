---

### 2. Criar os arquivos da API e do GitHub na sua pasta (5 min)
Na mesma pasta `Classificador_Avaliacao_Delivery`, crie estes 3 arquivos leves (substitua `SEU_USUARIO` no `app.py` e no `README.md`):

* **`app.py`** (mostra que você sabe transformar o modelo em um microsserviço REST para o time de *Engenharia de Software AI*):
  ```python
  from fastapi import FastAPI
  from transformers import pipeline

  app = FastAPI(title="API de Classificação de Reviews - Delivery")
  classifier = pipeline("text-classification", model="SEU_USUARIO/ifood-delivery-sentiment-bertimbau")

  @app.post("/analisar")
  def analisar_review(texto: str):
      resultado = classifier(texto)[0]
      return {
          "review": texto,
          "classificacao": resultado["label"],
          "confianca": round(resultado["score"], 4)
      }