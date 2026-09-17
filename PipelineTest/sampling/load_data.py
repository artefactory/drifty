from datasets import load_dataset



def load_triviaqa(num_samples=100, seed=42):
    ds = load_dataset("trivia_qa", "rc", split="train")
    subset = ds.shuffle(seed=seed).select(range(num_samples))
    data = []
    labels = []
    for item in subset:
        question = item['question']
        data.append([{"role": "user", 'content': question}])
        # Gold answer first (used as "Expected Answer" by eval.py), then aliases
        gold = item['answer']['value']
        aliases = [a for a in item['answer']['aliases'] if a != gold]
        labels.append({"question": question, "label": [gold] + aliases})
    return data, labels


def load_naturalquestion(num_samples=100, seed=42):
    parquet_dir = "/data/workspace/.cache/huggingface/hub/datasets--google-research-datasets--natural_questions/snapshots/e8103d566bef4154c2c12b17c6095ec5275840cc/default"
    ds = load_dataset("parquet", data_files=f"{parquet_dir}/train-*.parquet", split="train")
    subset = ds.shuffle(seed=seed)
    data = []
    labels = []
    for item in subset:
        if len(data) >= num_samples:
            break
        # Extract short answer texts from annotations
        aliases = []
        for sa in item['annotations']['short_answers']:
            for text in sa['text']:
                if text and text not in aliases:
                    aliases.append(text)
        if not aliases:
            continue
        question = item['question']['text']
        data.append([{"role": "user", 'content': question}])
        labels.append({"question": question, "label": aliases})
    return data, labels


def load_hotpotqa(num_samples=100, seed=42):
    ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    subset = ds.shuffle(seed=seed).select(range(num_samples))
    data = []
    labels = []
    for item in subset:
        question = item['question']
        answer = item['answer']
        data.append([{"role": "user", 'content': question}])
        labels.append({"question": question, "label": [answer]})
    return data, labels


