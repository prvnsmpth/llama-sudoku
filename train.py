import re
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
import os

DATA_FILE = os.environ.get('DATA_FILE', 'data/dataset.json')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
LORA_DIR = os.environ.get('LORA_DIR', 'lora')
MERGED_DIR = os.environ.get('MERGED_DIR', 'merged')
print("DATA_FILE:", DATA_FILE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("LORA_DIR:", LORA_DIR)
print("MERGED_DIR:", MERGED_DIR)

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 4096
lora_rank = 64

def train_grpo():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.8, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    # Load and prep dataset
    SYSTEM_PROMPT = """
    Your task is to solve the given 4x4 Sudoku puzzle. Given a puzzle, you will
    think step-by-step, and fill in the numbers in the grid.

    Respond in the following format:
    ```
    <think>
    ...
    </think>
    <answer>
    ...
    </answer>
    ```

    In the <think> tag, you should explain your reasoning.
    In the <answer> tag, you should only provide the well-formatted grid, as shown in the puzzle.
    """

    def dataset_formatter(example):
        puzzle = example['puzzle']
        solution = example['solution']
        return {
            'prompt': [
                { 'role': 'system', 'content': SYSTEM_PROMPT },
                { 'role': 'user', 'content': f'Solve this:\n{puzzle}' },
            ],
            'answer': f'{solution}'
        }

    dataset = load_dataset('json', data_files=DATA_FILE, split='train')
    dataset = dataset.map(lambda x: dataset_formatter(x))

    def parse_grid(grid_str) -> list[list[str]]:
        """Parse a grid like:
        +---+---+---+---+
        | 3 | 1 | 2 | 4 |
        +---+---+---+---+
        | 4 | . | 3 | 1 |
        +---+---+---+---+
        | 2 | . | 1 | . |
        +---+---+---+---+
        | 1 | 3 | 4 | 2 |
        +---+---+---+---+
        """
        if not grid_str:
            return None
        
        if not is_well_formatted_grid(grid_str):
            return None

        grid = []
        for line in grid_str.strip().split('\n'):
            if line.startswith('+'):
                continue
            grid.append([c for c in line if c.isdigit() or c == '.'])
        
        if len(grid) != 4:
            return None
        for row in grid:
            if len(row) != 4:
                return None

        return grid

    def is_well_formatted_grid(grid_str) -> bool:
        """Check if the grid is well formatted, including the borders and the digits."""
        lines = grid_str.strip().split('\n')
        digit_line_pattern = re.compile(r'\| [1-4.] \| [1-4.] \| [1-4.] \| [1-4.] \|')
        if len(lines) != 9:
            return False
        for idx, line in enumerate(lines):
            line = line.strip()
            if len(line) != 17:
                return False
            
            if idx % 2 == 0:
                if line != '+---+---+---+---+':
                    return False
            else:
                if not digit_line_pattern.match(line):
                    return False

    def extract_xml_answer(text: str) -> str:
        if '<answer>' not in text or '</answer>' not in text:
            return None
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        answers = [parse_grid(a) for a in answer]
        extracted_responses = [parse_grid(extract_xml_answer(r)) for r in responses]
        reward = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answers)]
        print(f'correctness_reward_func: {reward[0]}')
        return reward

    def score_answer(solution: list[list[str]], response: list[list[str]]) -> float:
        if response is None:
            return 0.0
        num_matches = 0
        for i in range(4):
            for j in range(4):
                if solution[i][j] == response[i][j]:
                    num_matches += 1
        num_correct = max(0, num_matches - 8) # Subtract the number of clues
        return num_correct / 8.0

    def partial_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        answers = [parse_grid(a) for a in answer]
        extracted_responses = [parse_grid(extract_xml_answer(r)) for r in responses]
        reward = [score_answer(a, r) for r, a in zip(extracted_responses, answers)]
        print(f"partial_correctness_reward_func: {reward[0]}")
        return reward

    PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    STRICT_PATTERN = re.compile(r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$", re.DOTALL)

    def grid_format_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        reward = [1.0 if parse_grid(r) is not None else 0.0 for r in extracted_responses]
        print(f'grid_format_reward_func: {reward[0]}')
        return reward

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        responses = [completion[0]["content"] for completion in completions]
        matches = [STRICT_PATTERN.match(r) for r in responses]
        reward = [0.5 if match else 0.0 for match in matches]
        print(f'strict_format_reward_func: {reward[0]}')
        return reward

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        responses = [completion[0]["content"] for completion in completions]
        matches = [PATTERN.match(r) for r in responses]
        reward = [0.5 if match else 0.0 for match in matches]
        print(f'soft_format_reward_func: {reward[0]}')
        return reward

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            num_chars_beyond_answer = len(text.split("\n<answer>\n")[-1]) - 162
            penalty = max(0, num_chars_beyond_answer * 0.001)
            count -= penalty
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    def xmlcount_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [parse_grid(extract_xml_answer(r)) for r in responses]
        print('--' * 20)
        print('Num completions:', len(completions))
        print('Num responses:', len(responses))
        print(f"(Sample) Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        reward = [count_xml(c) for c in responses]
        print(f"xmlcount_reward_func: {reward[0]}")
        return reward

    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 4096,
        num_train_epochs = 5, # Set to 1 for a full training run
        max_steps = 1500,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = OUTPUT_DIR,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            grid_format_reward_func,
            partial_correctness_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()

    model.save_lora(LORA_DIR)
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method='merged_4bit')
    # model.save_pretrained_gguf("grpo_gguf_model_07032025094200", tokenizer, quantization_method = "q4_k_m")
