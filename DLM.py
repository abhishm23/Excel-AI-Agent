from gpt4all import GPT4All

# List of good lightweight models:
# - mistral-7b-instruct-v0.1.Q4_0.gguf
# - phi-2.Q4_0.gguf
# - or GPT4All-J, Falcon, etc.

model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
gpt = GPT4All(model_name)
gpt.download_model()