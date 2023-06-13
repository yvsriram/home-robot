import json

data = {}
raw_data = [
    "memory_test/memory_data_scene1.json",
    "memory_test/memory_data_scene2.json",
]

for raw_file in raw_data:
    raw_memory = json.load(raw_file)
