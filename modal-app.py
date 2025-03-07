import modal
from pathlib import Path

app = modal.App('llama-sudoku')
volume = modal.Volume.from_name('llama-sudoku-volume', create_if_missing=True)
VOLUME_MNT_PATH = "/vol"
TAG = "202503071040"
OUTPUT_DIR = Path(VOLUME_MNT_PATH, 'llama-sudoku', f"output_{TAG}")
LORA_DIR = Path(VOLUME_MNT_PATH, 'llama-sudoku', f"lora_{TAG}")
MERGED_DIR = Path(VOLUME_MNT_PATH, 'llama-sudoku', f"merged_{TAG}")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
        .apt_install("git")
        .run_commands("git clone https://github.com/prvnsmpth/llama-sudoku.git && echo 'Repo cloned.'", force_build=True)
        .pip_install([
            "unsloth",
            "vllm",
            "wandb"
        ])
        .workdir("llama-sudoku")
        .run_commands([
            "mkdir data",
            "python generate.py 50000 data/dataset.json",
            "wandb login 861c93a6f62db593d420d6dd0fb619c73b4a0c81",
        ])
        .env({
            'DATA_FILE': 'data/dataset.json',
            'OUTPUT_DIR': str(OUTPUT_DIR),
            'LORA_DIR': str(LORA_DIR),
            'MERGED_DIR': str(MERGED_DIR),
        })
)

@app.function(gpu='H100', image=image, volumes={ VOLUME_MNT_PATH: volume }, timeout=86_400)
def train():
    print("[Remote] Starting training...")
    from train import train_grpo
    train_grpo()

@app.local_entrypoint()
def main():
    print("[Local] Starting training...")
    train.remote()



