import json
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, dataset: str, **kwargs):
        if dataset == "chattime":
            self.loader = ChatTimeLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset: {dataset} (currently only `chattime`)")

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        return self.loader[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ChatTimeLoader:
    def __init__(self, test_path, vali_path=None, num_fewshot=3):
        self.items = self._load_data(test_path)
        
        # few-shot
        self.fewshot_map = {}
        if vali_path:
            vali_items = self._load_data(vali_path)
            for item in vali_items:
                task = item.get('task')
                if task not in self.fewshot_map:
                    self.fewshot_map[task] = []
                if len(self.fewshot_map[task]) < num_fewshot:
                    self.fewshot_map[task].append(item)

    def _load_data(self, path):
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                data = [json.loads(l) for l in f if l.strip()]
            else:
                data = json.load(f)
                
        return data

    def __getitem__(self, idx):
        item = self.items[idx]
        task = item['task']
        
        # few-shot prompt construction
        example_prompt = ["Below are some examples of time series analysis tasks:\n"]
        for i, ex in enumerate(self.fewshot_map.get(task, []), 1):
            example_prompt.append(f"Example {i}:\nInput: Data={ex['timeseries']}, Length={len(ex['timeseries'])}")
            example_prompt.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n")
        
        # add the test question
        full_question = [
            f"\n".join(example_prompt),
            f"\nNow solve the following task:\n\nAct as an expert time series data scientist. Input: Data={item['timeseries']}, Length={len(item['timeseries'])}\n\nQuestion: {item['question']}\n",
            f"Answer:"
        ]

        full_question = "".join(full_question)
        question = item['question']
        answer = item['answer']
        
        return {
            "full_question": full_question,
            "question": question,
            "answer": answer,
            "timeseries": item['timeseries'],
            "metadata": {
                "task": task, 
                "size": item.get('size', ''),
                'label': item.get('label', ''),
                'timeseries_length': len(item['timeseries'])
            }
        }

    def __len__(self):
        return len(self.items)
