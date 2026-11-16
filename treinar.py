import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

#Dataset de exemplo (Avaliações de Delivery)
dados = {
    "text": [
        "A comida chegou muito fria e o refrigerante derramou na sacola.",
        "Entregador super educado e o pedido chegou antes do prazo!",
        "O lanche estava delicioso, bem embalado e quentinho.",
        "Demorou quase duas horas para entregar, péssima experiência.",
        "Faltou um item do meu combo e o restaurante não responde no chat.",
        "Comida sem tempero e batata murcha, não peço mais.",
        "Entrega rápida demais, parabéns pela agilidade!",
        "Melhor sushi que já pedi pelo app, tudo fresco e saboroso.",
        "O motoboy se perdeu e deixou o pedido na portaria errada.",
        "Veio tudo errado, pedi sem cebola e veio cheio.",
        "Preço justo, porção bem servida e entrega pontual.",
        "A embalagem veio violada e o pedido atrasou 50 minutos.",
        "Excelente atendimento do restaurante, mandaram até brinde!",
        "A pizza chegou toda revirada dentro da caixa por causa do transporte.",
        "Sabor incrível, temperatura perfeita, virei cliente fiel.",
        "Atendimento horrível e demora absurda na preparação."
    ] * 4,
    "label": [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 4
}

dataset = Dataset.from_dict(dados).train_test_split(test_size=0.2, seed=42)

#Tokenizer e Modelo
model_checkpoint = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

id2label = {0: "NEGATIVO_PROBLEMA", 1: "POSITIVO_ELOGIO"}
label2id = {"NEGATIVO_PROBLEMA": 0, "POSITIVO_ELOGIO": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

#Métricas
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#Treino e Push para o Hugging Face
nome_repo_hf = "food-delivery-sentiment-bertimbau"

training_args = TrainingArguments(
    output_dir=nome_repo_hf,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    push_to_hub=False,
    use_cpu=True,
    hub_model_id=nome_repo_hf
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
print("Modelo publicado com sucesso no Hugging Face!")