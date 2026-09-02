from __future__ import annotations

from scripts.create_balanced_multitask_dataset import select_task_episodes


def test_task_selection_is_balanced_disjoint_and_deterministic() -> None:
    episodes = [
        {"file_name": f"episode_{index:04d}.hdf5"}
        for index in range(100)
    ]

    first_train, first_val = select_task_episodes(
        episodes,
        task="box",
        total=30,
        val_count=3,
        seed=42,
    )
    second_train, second_val = select_task_episodes(
        episodes,
        task="box",
        total=30,
        val_count=3,
        seed=42,
    )

    train_names = {item["file_name"] for item in first_train}
    val_names = {item["file_name"] for item in first_val}
    assert len(train_names) == 27
    assert len(val_names) == 3
    assert train_names.isdisjoint(val_names)
    assert [item["file_name"] for item in first_train] == [
        item["file_name"] for item in second_train
    ]
    assert [item["file_name"] for item in first_val] == [
        item["file_name"] for item in second_val
    ]


def test_task_name_changes_selection_stream() -> None:
    episodes = [
        {"file_name": f"episode_{index:04d}.hdf5"}
        for index in range(100)
    ]
    box_train, _ = select_task_episodes(
        episodes,
        task="box",
        total=30,
        val_count=3,
        seed=42,
    )
    bottle_train, _ = select_task_episodes(
        episodes,
        task="bottle",
        total=30,
        val_count=3,
        seed=42,
    )
    assert [item["file_name"] for item in box_train] != [
        item["file_name"] for item in bottle_train
    ]
