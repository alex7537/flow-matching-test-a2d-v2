from __future__ import annotations


EXECUTED_ACTION_SEMANTICS = "executed_joint_position"
HYBRID_ACTION_SEMANTICS = "arm_executed_hand_commanded_joint_position"

ACTION_LAYOUTS = {
    EXECUTED_ACTION_SEMANTICS: "arm2_pos(7)+hand2_pos(6)",
    HYBRID_ACTION_SEMANTICS: "arm2_pos(7)+hand2_pos_target(6)",
}

ACTION_COMPONENTS = {
    EXECUTED_ACTION_SEMANTICS: [
        {"name": "arm2_pos", "dim": 7},
        {"name": "hand2_pos", "dim": 6},
    ],
    HYBRID_ACTION_SEMANTICS: [
        {"name": "arm2_pos", "dim": 7},
        {"name": "hand2_pos_target", "dim": 6},
    ],
}


def validate_action_semantics(value: str) -> str:
    semantics = str(value)
    if semantics not in ACTION_LAYOUTS:
        supported = ", ".join(sorted(ACTION_LAYOUTS))
        raise ValueError(f"unsupported action_semantics={semantics!r}; expected one of: {supported}")
    return semantics


def action_layout(semantics: str) -> str:
    return ACTION_LAYOUTS[validate_action_semantics(semantics)]


def action_components(semantics: str) -> list[dict[str, int | str]]:
    return [dict(item) for item in ACTION_COMPONENTS[validate_action_semantics(semantics)]]
