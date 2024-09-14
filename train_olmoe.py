import os
import yaml
import wandb
import torch
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
from transformers import AutoConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"


MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
auto_config = AutoConfig.from_pretrained(MODEL_NAME)
layer = 7
total_training_steps = 5_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
expansion_factor = 8
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def train(config=None):
    with wandb.init(config=config):

        config = wandb.config
        learning_rate = config.learning_rate
        l1_coefficient = config.l1_coefficient

        cfg = LanguageModelSAERunnerConfig(
            model_name=MODEL_NAME,
            hook_name=f"blocks.{layer}.hook_resid_pre",
            hook_layer=layer,
            d_in=auto_config.hidden_size,
            dataset_path="allenai/OLMoE-mix-0924",
            is_dataset_tokenized=False,
            streaming=True,
            mse_loss_normalization=None,
            expansion_factor=expansion_factor,
            b_dec_init_method="zeros",
            apply_b_dec_to_input=False,
            normalize_sae_decoder=False,
            scale_sparsity_penalty_by_decoder_norm=True,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
            normalize_activations="expected_average_only_in",
            # Training Parameters
            lr=learning_rate,
            lr_scheduler_name="constant",
            lr_warm_up_steps=lr_warm_up_steps,
            lr_decay_steps=lr_decay_steps,
            l1_coefficient=l1_coefficient,
            l1_warm_up_steps=l1_warm_up_steps,
            train_batch_size_tokens=batch_size,
            context_size=128,  # will control the length of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
            # Activation Store Parameters
            n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
            training_tokens=total_training_tokens,
            store_batch_size_prompts=16,
            # WANDB
            log_to_wandb=True,
            wandb_project=f"SAE-olmoe-1b-7b-0924-layer{layer}",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
            # Misc
            device=device,
            n_checkpoints=10,
            checkpoint_path=f"checkpoints-layer{layer}",
            dtype="bfloat16",
        )

        sae = SAETrainingRunner(cfg).run()
        sae.save_model(f"trained-layer{layer}")

if __name__ == "__main__":
    with open("./olmoe-config.yaml") as file:
        train(yaml.load(file, Loader=yaml.FullLoader))
