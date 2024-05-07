from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.111.132.198:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

if __name__ == "__main__":
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user",
             "content": "你是谁"}
        ],
        temperature=0.3,
        stream=True
    )
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
