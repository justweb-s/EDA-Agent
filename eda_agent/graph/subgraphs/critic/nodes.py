"""Critic nodes."""

from __future__ import annotations

from typing import Any, cast

from eda_agent.graph.subgraphs.critic.state import CriticState
from eda_agent.models.critic import CriticReview
from eda_agent.models.notebook import NotebookCell


def read_section(*args: Any, **kwargs: Any) -> Any:
    state = cast(CriticState, args[0] if args else kwargs.get("state"))
    section_name = str(state.get("section_name") or "all")

    notebook_cells_raw = list(state.get("notebook_cells", []))
    notebook_cells: list[NotebookCell] = [
        (c if isinstance(c, NotebookCell) else NotebookCell.model_validate(c))
        for c in notebook_cells_raw
    ]

    if section_name == "all":
        return {"cells_analyzed": notebook_cells}

    wanted_prefix = f"## {section_name}:"
    wanted_step_ids: set[str] = set()
    for cell in notebook_cells:
        if cell.cell_type != "markdown":
            continue
        if cell.source.strip().startswith(wanted_prefix) and cell.step_id:
            wanted_step_ids.add(cell.step_id)

    if not wanted_step_ids:
        return {"cells_analyzed": []}

    section_cells = [c for c in notebook_cells if (c.step_id or "") in wanted_step_ids]
    return {"cells_analyzed": section_cells}


def evaluate_section(*args: Any, **kwargs: Any) -> Any:
    state = cast(CriticState, args[0] if args else kwargs.get("state"))
    section_name = str(state.get("section_name") or "all")
    cells_raw = list(state.get("cells_analyzed", []))
    cells: list[NotebookCell] = [
        (c if isinstance(c, NotebookCell) else NotebookCell.model_validate(c)) for c in cells_raw
    ]

    code_cells = [c for c in cells if c.cell_type == "code"]
    code_with_outputs = [c for c in code_cells if c.outputs]
    markdown_cells = [c for c in cells if c.cell_type == "markdown"]

    missing_analyses: list[str] = []
    shallow_interpretations: list[str] = []
    proposed_steps: list[str] = []

    if not cells:
        missing_analyses.append(f"No cells found for section '{section_name}'.")
    if not code_cells:
        missing_analyses.append("No executed code cells were produced.")

    short_markdowns = [c for c in markdown_cells if len(c.source.strip()) < 30]
    if short_markdowns and section_name != "all":
        shallow_interpretations.append("Section markdown context appears too short.")

    denom = max(1, len(code_cells))
    ratio = len(code_with_outputs) / denom
    quality_score = 0.4 + 0.6 * min(1.0, ratio)
    if missing_analyses:
        quality_score = min(quality_score, 0.45)

    verdict: str
    if quality_score < 0.5:
        verdict = "needs_improvement"
        if section_name != "all":
            proposed_steps.append(
                f"Add at least one additional analysis step for section '{section_name}'."
            )
    else:
        verdict = "ok"

    review = CriticReview(
        quality_score=float(round(quality_score, 3)),
        verdict=cast(Any, verdict),
        missing_analyses=missing_analyses,
        shallow_interpretations=shallow_interpretations,
        proposed_steps=proposed_steps,
    )
    return {"critic_feedback": review}
