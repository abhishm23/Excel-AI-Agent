from gpt4all import GPT4All

# Instantiate model
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="models", allow_download=False)

# Test prompt
prompt = (
    "You are a financial analyst. Based on the data below, write a one-line insight.\n\n"
    "Revenue increased from 1000 $ to 1200 $. Profit rose from 200 $ to 300 $.\n\n"
    "Insight:"
)

# Run generation
with model.chat_session():
    response = model.generate(prompt, max_tokens=100, temp=0.7)

print("\n📣 AI Commentary:\n", response)
