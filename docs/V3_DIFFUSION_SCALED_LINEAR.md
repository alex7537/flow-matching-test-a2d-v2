# V3 Diffusion Policy: scaled-linear baseline

This experiment holds the V3 RGB+proprio data, hybrid action semantics, ViT,
Transformer, optimizer-step budget, augmentation, seed, EMA, and checkpoint
selection contract fixed against the successful CFM baseline. It changes only
the action-generation family and its intrinsic sampler parameters.

## Forward process

CFM uses a continuous linear interpolation:

```text
x_t = (1-t) noise + t action
```

This Diffusion Policy instead corrupts the clean normalized action with:

```text
x_k = sqrt(alpha_bar_k) action + sqrt(1-alpha_bar_k) epsilon
```

The beta values increase linearly after scaling the standard 1000-step
`1e-4 -> 2e-2` range to 100 training timesteps:

```text
beta_k = linear(0.001, 0.2)
terminal alpha_bar ~= 2.04e-5
terminal signal coefficient ~= 0.00452
```

The endpoint is therefore close to pure noise. Using an unscaled
`1e-4 -> 2e-2` range for only 100 timesteps would retain too much signal.

## Correctness fixes before launch

- deterministic per-sample validation randomness for DP and RS-IMLE;
- one observation encoding per iterative CFM/DDIM sample;
- timestep-bucket epsilon losses;
- predicted-x0 MSE and pre-clamp saturation fraction;
- existing DDIM x0 clipping to the normalized action range `[-1,1]`.

These diagnostics do not change the epsilon-MSE training objective.

## Matched budget

```text
effective train samples  220,026
batch size               32
steps per epoch          6,876
epochs                   100
total optimizer steps    687,600
warmup steps             34,380
head peak LR             1e-4
ViT peak LR              1e-5
```

Training loss must only be compared within DP runs. Final comparison with CFM
uses the same strict Isaac rollout protocol, seeds, execute horizon, and
success/retention criteria.
