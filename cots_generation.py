import torch
import time
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import cot_to_list

MODEL_ID = "/mnt/qwen"
SAVE_DIR = "/mnt/saves"
SYSTEM_PROMPT = "You are a creative assistant."

def generate_batch(prompt, num_generations, model, tokenizer, text_generator, temperature=0.8, batch_size=16):
    """
    Generates a batch of answers given a single prompt using HuggingFace's pipeline.

    Args:
        prompt (str): The user prompt to pass to the model.
        num_generations (int): The total number of CoTs to generate.
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        text_generator (Pipeline): The initialized 'text-generation' pipeline.
        temperature (float, optional): The sampling temperature. Defaults to 0.8.
        batch_size (int, optional): The batch size for inference. Defaults to 16.

    Returns:
        list: A list of strings containing only the generated response text.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    prompt_batch = [formatted_prompt] * num_generations

    outputs = text_generator(
        prompt_batch,
        max_new_tokens=1536,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=batch_size
    )

    responses_only = []
    for result_list in outputs:
        full_text = result_list[0]['generated_text']
        response = full_text[len(formatted_prompt):].strip()
        responses_only.append(response)
    
    return responses_only

def main():
    """
    Main entry point. Loads the model, executes batch generation of CoTs 
    for the specified problem, evaluates the answers, and saves the results as JSON.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa"
    )
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    n_cots = 32
    filename = "cots_precalculus_lv5_7"
    solution = r'''\frac{1}{\sqrt{2}}'''
    problem = (
        r"There is an angle $\theta$ in the range $0^\circ < \theta < 45^\circ$ which satisfies"
        "\n\\[\tan \theta + \tan 2 \theta + \tan 3 \theta = 0.\\]Calculate $\\tan \\theta$ for this angle."
    )

    pre_prompt = (
        "Let's solve the following math problem step by step.\n\n"
        "**Required Formatting:**\n"
        "* Provide a step-by-step solution.\n"
        "* Each step **must** begin on a new line and be formatted *exactly* as `Step N:` (e.g., `Step 1:`, `Step 2:`, etc.).\n"
        "* The final section **must** begin on a new line and be formatted *exactly* as `Final Answer:`.\n"
        "* The final mathematical answer **must** be enclosed in the `\\boxed{}` command.\n\n"
        "**Problem:**\n"
    )
    prompt = pre_prompt + problem

    start_time = time.time()
    all_answers_text = generate_batch(prompt, n_cots, model, tokenizer, text_generator, temperature=0.85, batch_size=32)
    
    cots = []
    for answer_text in all_answers_text:
        cot = cot_to_list(answer_text)
        if cot and cot[-1]:
            cots.append(cot)

    exec_time = time.time() - start_time
    minutes, seconds = divmod(exec_time, 60)
    print(f"\nGeneration time for {n_cots} CoTs: {int(minutes)} minutes and {int(seconds)} seconds")

    final_answers = [cot[-1] for cot in cots]
    answer_counts = Counter(final_answers)
    solution_generated = solution in final_answers

    sorted_answers = sorted(answer_counts.items(), key=lambda item: (item[0] != solution, -item[1]))

    output_data = {
        "problem": problem,
        "solution": solution,
        "solution_generated": solution_generated,
        "sorted_answers": sorted_answers,
        "cots": cots
    }

    json_dest_path = f"{SAVE_DIR}/{filename}.json"
    with open(json_dest_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"\nData saved in JSON format at: {json_dest_path}")

if __name__ == "__main__":
    main()