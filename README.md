I'm now focused on Olmoe, though `train_mixtral.py` still exists.

Sweep:

```
wandb sweep --project moe-sae olmoe-config.yaml
wandb agent <sweep id printed by previous command>
```

Train:

```
python3 train_olmoe.py
```
