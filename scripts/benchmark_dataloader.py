from __future__ import annotations

import argparse
import json
import time

from torch.utils.data import DataLoader

from flow_matching_test.train import _build_dataset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark processed RGB dataloader throughput")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    cfg["data"]["data_dir"] = args.data_dir
    dataset = _build_dataset(data_cfg=cfg["data"], split="train", seed=int(cfg["training"]["seed"]))
    batch_size = int(cfg["training"]["batch_size"])
    results = []
    for workers in args.workers:
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(**loader_kwargs)
        start = time.perf_counter()
        samples = batches = 0
        for batch in loader:
            samples += int(batch["action"].shape[0])
            batches += 1
            if batches >= args.batches:
                break
        elapsed = time.perf_counter() - start
        results.append({
            "workers": workers,
            "batches": batches,
            "samples": samples,
            "elapsed_sec": elapsed,
            "samples_per_sec": samples / elapsed,
            "rgb_frames_per_sec": samples * len(cfg["data"]["image_keys"]) / elapsed,
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
