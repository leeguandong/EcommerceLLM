import json
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://10.111.132.198:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


# Function to evaluate each instruction
def evaluate(instruction):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instruction}
        ],
        temperature=0.3,
        stream=True
    )
    content = []
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
            content.append(chunk.choices[0].delta.content)
    return "".join(content)


# Read the input file
with open(r"E:\comprehensive_library\e_commerce_llm\evaluation\general_test.json", "r", encoding='UTF-8') as input_file:
    input_data = [json.loads(line) for line in input_file]

results = []

# Evaluate each line
for idx, data in enumerate(input_data):
    output = evaluate(data['instruction'])
    results.append({'instruction': data['instruction'], 'input': data['input'], 'output': output})

# Write the output file
with open("E:/comprehensive_library/e_commerce_llm/results/output_llama3_8B_base_sft.json", "w", encoding='UTF-8') as outfile:
    for line in results:
        json.dump(line, outfile, ensure_ascii=False)
        outfile.write("\n")
