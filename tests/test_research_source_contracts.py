from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SOURCE = REPO_ROOT / "src" / "earnings_event_vol" / "research.py"


def _range_bound(node: ast.AST, index: int) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "XGBOOST_MIN_CHILD_WEIGHT_RANGE"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == index
    )


def test_xgboost_min_child_weight_uses_continuous_search_space() -> None:
    tree = ast.parse(RESEARCH_SOURCE.read_text(encoding="utf-8"))
    min_child_range: tuple[float, float] | None = None
    xgboost_param_function: ast.FunctionDef | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "XGBOOST_MIN_CHILD_WEIGHT_RANGE"
            and isinstance(node.value, ast.Tuple)
        ):
            range_values = [
                float(element.value)
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, (int, float))
            ]
            if len(range_values) == 2:
                min_child_range = (range_values[0], range_values[1])
        if isinstance(node, ast.FunctionDef) and node.name == "_xgboost_trial_params":
            xgboost_param_function = node

    assert min_child_range == (3.0, 50.0)
    assert xgboost_param_function is not None

    float_call_found = False
    categorical_call_found = False
    for call_node in ast.walk(xgboost_param_function):
        if not (
            isinstance(call_node, ast.Call)
            and isinstance(call_node.func, ast.Attribute)
            and call_node.args
            and isinstance(call_node.args[0], ast.Constant)
            and call_node.args[0].value == "min_child_weight"
        ):
            continue
        if call_node.func.attr == "suggest_float":
            float_call_found = (
                len(call_node.args) >= 3
                and _range_bound(call_node.args[1], 0)
                and _range_bound(call_node.args[2], 1)
            )
        if call_node.func.attr == "suggest_categorical":
            categorical_call_found = True

    assert float_call_found
    assert categorical_call_found is False
