import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_lens import LanguageModelSAERunnerConfig, language_model_sae_runner
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"

pretrained_model_config = get_pretrained_model_config(MODEL_NAME)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name=MODEL_NAME,
    hook_point="blocks.{layer}.hook_mlp_out",
    # hook_point_layer=2,
    d_in=pretrained_model_config.d_mlp,  # d_model
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    # SAE Parameters
    expansion_factor=[2, 4, 8, 16, 32],
    b_dec_init_method="geometric_median",
    # Training Parameters
    l1_coefficient=[1e-3, 2.5e-3, 5e-3, 1e-2],
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    # WANDB
    log_to_wandb=True,
    wandb_project=f"SAE-{MODEL_NAME}",
    wandb_log_frequency=100,
    # Misc
    device=device,
    n_checkpoints=10,
    dtype=torch.float32,
)

sparse_autoencoder = language_model_sae_runner(cfg)
