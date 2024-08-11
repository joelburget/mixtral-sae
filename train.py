import os

import torch
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from Auto_HookPoint import (
    HookedTransformerAdapter,
    HookedTransformerAdapterCfg,
    HookedTransformerConfig_From_AutoConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"


MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"

auto_config = AutoConfig.from_pretrained(MODEL_NAME)

total_training_steps = 50_000
batch_size = 8
total_training_tokens = total_training_steps * batch_size
expansion_factor = 8
lr = 4e-5
lp_norm = 1.0

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def preprocess_mixtral(model: HookedTransformer, input):
    if isinstance(input, (str, list)):
        tokens = model.to_tokens(input).to(model.cfg.device)
    else:
        tokens = input.to(model.cfg.device)
    return tokens, model.hook_embed(model.embed(tokens))


adapter_cfg = HookedTransformerAdapterCfg(
    mappings={
        "blocks": "model.layers",
        "unembed": "lm_head",
        "embed": "model.embed_tokens",
        "pos_embed": None,  # DUMMY
        "ln_final": "model.norm",
    },
    inter_block_fn=lambda x: x[0],
    create_kwargs=lambda cfg, residual: {
        "position_ids": torch.arange(residual.shape[1], device=residual.device).expand(
            residual.shape[0], -1
        )
    },
    preprocess=preprocess_mixtral,
)

hooked_transformer_cfg = HookedTransformerConfig_From_AutoConfig.from_auto_config(
    auto_config,
    attn_only=True,
    normalization_type=None,
    positional_embedding_type="rotary",
)


if __name__ == "__main__":
    for layer in [5, 16, 27]:
        cfg = LanguageModelSAERunnerConfig(
            hook_name=f"model.layers.{layer}.post_attention_layernorm.hook_point",
            hook_layer=layer,
            d_in=auto_config.hidden_size,
            dataset_path="monology/pile-uncopyrighted",
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
            lr=lr,
            adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
            adam_beta2=0.999,
            lr_scheduler_name="constant",
            lr_warm_up_steps=lr_warm_up_steps,
            lr_decay_steps=lr_decay_steps,
            l1_coefficient=5,
            l1_warm_up_steps=l1_warm_up_steps,
            lp_norm=lp_norm,
            train_batch_size_tokens=batch_size,
            context_size=128,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
            # Activation Store Parameters
            n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
            training_tokens=total_training_tokens,
            store_batch_size_prompts=16,
            # Resampling protocol
            use_ghost_grads=False,
            feature_sampling_window=1000,
            dead_feature_window=1000,
            dead_feature_threshold=1e-4,
            # WANDB
            log_to_wandb=True,
            wandb_project=f"SAE-mixtral-8x7b-v0.1-layer{layer}",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
            # Misc
            device=device,
            n_checkpoints=10,
            checkpoint_path=f"checkpoints-layer{layer}",
            dtype="float32",
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="balanced_low_0"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        hooked_model = HookedTransformerAdapter(
            adapter_cfg=adapter_cfg,
            hooked_transformer_cfg=hooked_transformer_cfg,
            model=model,
            tokenizer=tokenizer,
        )

        cfg.device = device
        sae = SAETrainingRunner(cfg, override_model=hooked_model).run()
        sae.save_model(f"trained-layer{layer}")
