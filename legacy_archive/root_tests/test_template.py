from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("models/science_v2")
print("Chat template exists:", tokenizer.chat_template is not None)
if tokenizer.chat_template:
    print(tokenizer.chat_template[:200])

messages = [{"role": "user", "content": "Hello"}]
try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Formatted prompt:")
    print(repr(prompt))
except Exception as e:
    print("Failed to apply chat template:", e)
