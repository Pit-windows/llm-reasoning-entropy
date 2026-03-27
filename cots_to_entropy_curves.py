import torch
import time
import json
import numpy as np
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/mnt/qwen-8b-math"
SAVE_DIR = "/mnt/saves"

def calculate_conditional_entropy_batched(contexts_c, response_y, model, tokenizer):
    """
    Computes the conditional entropy H(Y|C) for a batch of contexts and a single response.
    Uses left padding for efficient batch processing and aligns tokens accurately.

    Args:
        contexts_c (list): A list of string contexts (the reasoning steps up to a certain point).
        response_y (str): The target response string.
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.

    Returns:
        list: A list of float values representing the mean entropy for each context.
    """
    if not contexts_c:
        return []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    response_tokens = tokenizer.encode(" " + response_y, add_special_tokens=False)
    response_len = len(response_tokens)
    
    if response_len == 0:
        return [0.0] * len(contexts_c)

    inputs = [f"{c} {response_y}" for c in contexts_c]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
    
    input_ids = tokenized_inputs.input_ids
    attention_mask = tokenized_inputs.attention_mask
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    seq_len = input_ids.shape[1]
    start_logits_idx = seq_len - response_len - 1
    end_logits_idx = seq_len - 1
    relevant_logits = logits[:, start_logits_idx:end_logits_idx, :]

    entropies = Categorical(logits=relevant_logits).entropy()
    mean_entropies = entropies.mean(dim=1)
    
    return mean_entropies.tolist()

def calculate_entropy_data(cots, target_answer, filename, model, tokenizer):
    """
    Computes entropy curves given a list of CoTs with respect to a target answer. 
    Uses batching, aggregating contexts from different CoTs and parallelizing computation.

    Args:
        cots (list): A list of Chain-of-Thought reasoning paths.
        target_answer (str): The final answer string to evaluate against.
        filename (str): The output JSON filepath to save the curves.
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
    """
    start = time.time()
    max_steps = max((len(cot) for cot in cots), default=0)
    contexts_by_step = [[] for _ in range(max_steps)]
    
    for i, cot in enumerate(cots):
        for step in range(1, len(cot)):
            contexts_by_step[step].append({
                "cot_idx": i, 
                "context": " ".join(cot[:step])
            })

    entropy_curves_map = {i: [] for i in range(len(cots))}

    for step_data in contexts_by_step:
        if not step_data:
            continue
            
        contexts = [d["context"] for d in step_data]
        cot_indices = [d["cot_idx"] for d in step_data]
        
        entropies = calculate_conditional_entropy_batched(contexts, target_answer, model, tokenizer)
        
        for i, entropy in enumerate(entropies):
            entropy_curves_map[cot_indices[i]].append(entropy)

    results_list = [
        {"curve": np.array(entropy_curves_map[i]).tolist(), "original_answer": cot[-1]}
        for i, cot in enumerate(cots)
    ]

    print(f"'calculate_entropy_data' - {len(cots)} CoTs - Time: {time.time() - start:.4f} s")

    entropy_data = {"target_answer": target_answer, "results": results_list}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entropy_data, f, indent=4)
    print(f"Data saved to {filename}")

def main():
    """
    Main entry point. Loads the model, formats the generated CoTs, and computes 
    conditional entropy curves for the top candidate answers, saving them to JSON.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    problem = "int_algebra_lv4_2"
    source_filename = f"{SAVE_DIR}/cots_{problem}"
    dest_filename = f"{SAVE_DIR}/entropy_curves_{problem}"
    start_index = 0
    end_index = 1 

    with open(f"{source_filename}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cots = data["cots"]
    problem_text = data["problem"]

    for cot in cots:
        cot[0] = f"Let's solve this math problem step by step.\nProblem: {problem_text}\nSolution:\n{cot[0]}"

    for i in range(start_index, end_index):
        answer = data["sorted_answers"][i][0]
        dest_path = f"{dest_filename}_answer_{i}.json"
        calculate_entropy_data(cots, answer, dest_path, model, tokenizer)

if __name__ == "__main__":
    main()