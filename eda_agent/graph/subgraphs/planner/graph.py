"""Planner subgraph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from eda_agent.graph.coercion import coerce_dataset_context
from eda_agent.graph.subgraphs.planner.state import PlannerState
from eda_agent.models.plan import EDAStep


def build_planner_subgraph(*args: Any, **kwargs: Any) -> Any:
    graph = StateGraph(PlannerState)

    def draft_plan(state: PlannerState) -> PlannerState:
        dataset_context = coerce_dataset_context(state.get("dataset_context"))
        if dataset_context is None:
            return {
                "draft_plan": [],
                "validation_errors": ["Missing dataset_context"],
                "iteration_count": 1,
            }

        columns = dataset_context.columns
        numeric_cols = [
            c.name for c in columns if any(t in c.dtype.lower() for t in ["int", "float"])
        ][:3]
        categorical_cols = [
            c.name for c in columns if not any(t in c.dtype.lower() for t in ["int", "float"])
        ][:3]
        datetime_cols = [
            c.name for c in columns if c.detected_semantic_type in {"datetime", "date"}
        ][:2]

        draft: list[EDAStep] = []

        draft.append(
            EDAStep(
                step_id="data_quality",
                section="Data quality",
                title="Data quality checks",
                description="Review missing values, duplicates, and detected issues.",
                analysis_type="data_quality",
                target_columns=[],
                depends_on=[],
                is_mandatory=True,
                priority=10,
            )
        )

        if numeric_cols:
            draft.append(
                EDAStep(
                    step_id="univariate_numeric",
                    section="Univariate",
                    title="Numeric distributions",
                    description="Inspect distributions and summary statistics for numeric columns.",
                    analysis_type="univariate",
                    target_columns=numeric_cols,
                    depends_on=["data_quality"],
                    is_mandatory=True,
                    priority=8,
                )
            )

        if categorical_cols:
            draft.append(
                EDAStep(
                    step_id="univariate_categorical",
                    section="Univariate",
                    title="Categorical distributions",
                    description="Inspect value counts and rare categories for categorical columns.",
                    analysis_type="univariate",
                    target_columns=categorical_cols,
                    depends_on=["data_quality"],
                    is_mandatory=True,
                    priority=7,
                )
            )

        if len(numeric_cols) >= 2:
            draft.append(
                EDAStep(
                    step_id="bivariate_correlation",
                    section="Bivariate",
                    title="Correlation overview",
                    description=(
                        "Review correlations and pairwise relationships between numeric columns."
                    ),
                    analysis_type="bivariate",
                    target_columns=numeric_cols,
                    depends_on=["univariate_numeric"],
                    is_mandatory=False,
                    priority=5,
                )
            )

        if datetime_cols:
            draft.append(
                EDAStep(
                    step_id="feature_time",
                    section="Feature-specific",
                    title="Time-based patterns",
                    description="Inspect time trends and seasonality in datetime columns.",
                    analysis_type="feature_specific",
                    target_columns=datetime_cols,
                    depends_on=["data_quality"],
                    is_mandatory=False,
                    priority=4,
                )
            )

        return {
            "draft_plan": draft,
            "validation_errors": [],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }

    def finalize_plan(state: PlannerState) -> PlannerState:
        draft = list(state.get("draft_plan", []))
        errors = list(state.get("validation_errors", []))
        if not draft:
            errors.append("Planner produced an empty plan")
        return {
            "eda_plan": draft,
            "validation_errors": errors,
        }

    graph.add_node("draft", draft_plan)
    graph.add_node("finalize", finalize_plan)
    graph.set_entry_point("draft")
    graph.add_edge("draft", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
