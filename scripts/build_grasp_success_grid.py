#!/usr/bin/env python3
"""Build a reproducible 10x10 A2D grasp-success showcase."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


MODELS = (
    ("best_ema", "hand_target_best_ema"),
    ("epoch100_raw", "hand_target_epoch100_raw"),
    ("latest_raw", "hand_target_latest_raw"),
    ("rgb_only_final_raw", "rgb_only_final_raw"),
)
HORIZON_QUOTA = {4: 8, 8: 8, 16: 9}
HORIZON_COLORS = {4: "0x3b82f6", 8: "0x22c55e", 16: "0xf59e0b"}
NAME_RE = re.compile(
    r"^(?P<prefix>hand_target_best_ema|hand_target_epoch100_raw|"
    r"hand_target_latest_raw|rgb_only_final_raw)_h(?P<horizon>4|8|16)_"
    r"episode_(?P<episode>\d+)_seed_(?P<seed>\d+)_10cm\.mp4$"
)


@dataclass(frozen=True)
class Clip:
    path: Path
    model: str
    horizon: int
    episode: int
    seed: int


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def evenly_spaced(clips: list[Clip], count: int) -> list[Clip]:
    if len(clips) < count:
        raise ValueError(f"need {count} clips, found {len(clips)}")
    if count == 1:
        return [clips[len(clips) // 2]]
    indices = [round(index * (len(clips) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("even selection produced duplicate indices")
    return [clips[index] for index in indices]


def select_clips(source: Path) -> dict[str, list[Clip]]:
    by_category: dict[tuple[str, int], list[Clip]] = {}
    prefix_to_model = {prefix: model for model, prefix in MODELS}
    for path in sorted(source.glob("*_10cm.mp4")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        model = prefix_to_model[match.group("prefix")]
        horizon = int(match.group("horizon"))
        clip = Clip(
            path=path,
            model=model,
            horizon=horizon,
            episode=int(match.group("episode")),
            seed=int(match.group("seed")),
        )
        by_category.setdefault((model, horizon), []).append(clip)

    selected: dict[str, list[Clip]] = {}
    for model, _ in MODELS:
        model_clips: list[Clip] = []
        for horizon, quota in HORIZON_QUOTA.items():
            candidates = sorted(
                by_category.get((model, horizon), []),
                key=lambda clip: (clip.episode, clip.seed, clip.path.name),
            )
            model_clips.extend(evenly_spaced(candidates, quota))
        selected[model] = model_clips
    return selected


def grid_order(selected: dict[str, list[Clip]]) -> list[tuple[Clip, int, int]]:
    origins = {
        "best_ema": (0, 0),
        "epoch100_raw": (5, 0),
        "latest_raw": (0, 5),
        "rgb_only_final_raw": (5, 5),
    }
    ordered: list[tuple[Clip, int, int]] = []
    for model, _ in MODELS:
        origin_x, origin_y = origins[model]
        for index, clip in enumerate(selected[model]):
            ordered.append((clip, origin_x + index % 5, origin_y + index // 5))
    if len(ordered) != 100:
        raise ValueError(f"expected 100 selected clips, found {len(ordered)}")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=96)
    parser.add_argument("--tile-height", type=int, default=72)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    if not args.ffmpeg.is_file():
        raise SystemExit(f"ffmpeg does not exist: {args.ffmpeg}")

    selected = select_clips(source)
    ordered = grid_order(selected)
    for output in (args.gif, args.mp4, args.manifest):
        output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a2d-grasp-grid-") as temp_name:
        temp = Path(temp_name)
        prepared: list[Path] = []
        for index, (clip, _, _) in enumerate(ordered):
            output = temp / f"clip_{index:03d}.mp4"
            color = HORIZON_COLORS[clip.horizon]
            video_filter = (
                f"fps={args.fps},"
                f"scale={args.tile_width}:{args.tile_height},"
                f"drawbox=x=0:y=0:w=iw:h=ih:color={color}:t=2"
            )
            run(
                [
                    str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(args.start), "-i", str(clip.path),
                    "-t", str(args.duration), "-an", "-vf", video_filter,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                    "-pix_fmt", "yuv420p", str(output),
                ]
            )
            prepared.append(output)

        inputs: list[str] = []
        for path in prepared:
            inputs.extend(["-i", str(path)])
        layout = "|".join(
            f"{column * args.tile_width}_{row * args.tile_height}"
            for _, column, row in ordered
        )
        run(
            [
                str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                *inputs,
                "-filter_complex", f"xstack=inputs=100:layout={layout}:fill=black:shortest=1[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.mp4),
            ]
        )
        run(
            [
                str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(args.mp4),
                "-filter_complex",
                "[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop", "0", str(args.gif),
            ]
        )

    manifest = {
        "schema_version": 1,
        "source_collection": source.name,
        "selection": {
            "success_tier": "10cm",
            "models": {model: 25 for model, _ in MODELS},
            "horizons_per_model": {str(key): value for key, value in HORIZON_QUOTA.items()},
            "method": "evenly spaced by episode within each model/horizon category",
        },
        "render": {
            "layout": "10x10",
            "quadrants": {
                "top_left": "best_ema",
                "top_right": "epoch100_raw",
                "bottom_left": "latest_raw",
                "bottom_right": "rgb_only_final_raw",
            },
            "horizon_border_colors": HORIZON_COLORS,
            "start_seconds": args.start,
            "duration_seconds": args.duration,
            "fps": args.fps,
            "width": args.tile_width * 10,
            "height": args.tile_height * 10,
        },
        "clips": [
            {
                "cell": {"column": column, "row": row},
                "model": clip.model,
                "horizon": clip.horizon,
                "episode": clip.episode,
                "seed": clip.seed,
                "file": clip.path.name,
            }
            for clip, column, row in ordered
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
