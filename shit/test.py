from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "./models/llama3-8b-instruct"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True,  # important avec ta 4060 Ti 8 Go
)

pipe = pipeline("text-generation", model=model, tokenizer=tok)

out = pipe(
    "Écris un mini bilan psychomoteur fictif sur un enfant qui a des difficultées psychomoteurs avec du jargon médical. Fais  différentes sections claires comme Anamnèse synthétique, Bilan psychomoteur et Conclusion.",
    max_new_tokens=3000,
    temperature=0.4,
)
print(out[0]["generated_text"])
