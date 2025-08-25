import json
from tqdm import tqdm

with open('output-sd/qwen_image_scores.json', 'r') as file:
    json_data = json.load(file)

value = 0
count = 0
for item in tqdm(json_data):
    value += json_data[item][0]
    count += 1
print(value / count)