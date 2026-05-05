from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    implemented: bool
    justification: str
    risk: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "market_implied_event_variance": ModelSpec(
        model_id="market_implied_event_variance",
        role="baseline",
        implemented=True,
        justification="Uses IVAR_event as the market forecast to beat.",
        risk="Can dominate if earnings option prices are already efficient after costs.",
    ),
    "last_four_rvar": ModelSpec(
        model_id="last_four_rvar",
        role="baseline",
        implemented=False,
        justification="Same-ticker average realized variance over the prior four earnings events.",
        risk="No cross-sectional information and weak for regime changes.",
    ),
    "last_four_ivar": ModelSpec(
        model_id="last_four_ivar",
        role="baseline",
        implemented=False,
        justification=(
            "Same-ticker average implied event variance over the prior four earnings events."
        ),
        risk="Can inherit historical option-market bias.",
    ),
    "patell_wolfson_diagnostic": ModelSpec(
        model_id="patell_wolfson_diagnostic",
        role="diagnostic",
        implemented=False,
        justification=(
            "Patell-Wolfson-style diagnostic features based on pre-event implied-volatility "
            "behavior, realized earnings move history, and post-event volatility compression "
            "diagnostics; not a trainable model."
        ),
        risk="Should not be reported as a modern ML model.",
    ),
    "goyal_saretto_rv_iv_spread": ModelSpec(
        model_id="goyal_saretto_rv_iv_spread",
        role="feature_baseline",
        implemented=False,
        justification=(
            "RV-IV spread feature/baseline for event-level option ranking; not a full "
            "replication of the original portfolio design."
        ),
        risk="Useful as a cross-sectional benchmark, but it is not the paper's complete "
        "identification design.",
    ),
    "linear_elastic_net": ModelSpec(
        model_id="linear_elastic_net",
        role="model",
        implemented=False,
        justification="Transparent semi-structural tabular benchmark.",
        risk="Limited nonlinear interaction capacity.",
    ),
    "lightgbm": ModelSpec(
        model_id="lightgbm",
        role="model",
        implemented=False,
        justification="Strong tabular ML baseline before deep models.",
        risk="Does not directly encode pre-event sequences.",
    ),
    "ft_transformer": ModelSpec(
        model_id="ft_transformer",
        role="deep_model",
        implemented=False,
        justification="Deep tabular architecture for mixed numerical/categorical event features.",
        risk="May not beat GBDT on small tabular panels.",
    ),
    "mamba_sequence_encoder": ModelSpec(
        model_id="mamba_sequence_encoder",
        role="deep_model",
        implemented=False,
        justification="Selective sequence encoder for pre-event IV run-up and liquidity paths.",
        risk="The 20-day sequence may be too short for Mamba to add value.",
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    return MODEL_REGISTRY[model_id]


def unimplemented_model_message(model_id: str) -> str:
    spec = get_model_spec(model_id)
    if spec.implemented:
        return f"{model_id} is implemented as a deterministic baseline or diagnostic."
    return f"{model_id} is registered for the protocol but not implemented in v1."
