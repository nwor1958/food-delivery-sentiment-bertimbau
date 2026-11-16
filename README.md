# Classificador de Avaliações de Delivery com BERTimbau e FastAPI

Projeto de Engenharia de IA e PLN que realiza o *fine-tuning* de um modelo Transformer em português (`neuralmind/bert-base-portuguese-cased`) para triagem automatizada de feedbacks de clientes de delivery, publicado no **Hugging Face Hub** e exposto via API REST com **FastAPI**.

- **Modelo publicado no Hugging Face:** [nwor1958/food-delivery-sentiment-bertimbau](https://huggingface.co/nwor1958/food-delivery-sentiment-bertimbau)

## Estrutura do Projeto
- `treinar.py`: Pipeline de preparação de dados, tokenização, *fine-tuning* e publicação automática no Hugging Face Hub.
- `app.py`: API REST em FastAPI que consome o modelo diretamente do Hugging Face para inferência em tempo real.
- `requirements.txt`: Dependências do projeto.

## Como executar a API localmente
```bash
pip install -r requirements.txt
uvicorn app:app --reload