from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import nbformat

from eda_agent.models.notebook import CellOutput, NotebookCell


def export_ipynb_bytes(cells: Sequence[NotebookCell]) -> bytes:
    nb = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        }
    )

    nb_cells: list[Any] = []
    for cell in cells:
        if cell.cell_type == "markdown":
            nb_cells.append(nbformat.v4.new_markdown_cell(source=cell.source))
            continue

        outputs = [_to_nb_output(o) for o in cell.outputs]
        nb_cells.append(
            nbformat.v4.new_code_cell(
                source=cell.source,
                execution_count=cell.execution_count,
                outputs=outputs,
            )
        )

    nb["cells"] = nb_cells
    text = cast(str, nbformat.writes(nb))
    return text.encode("utf-8")


def _to_nb_output(output: CellOutput) -> Any:
    if output.output_type == "stream":
        return nbformat.v4.new_output(
            output_type="stream",
            name="stdout",
            text=output.text or "",
        )

    if output.output_type == "error":
        return nbformat.v4.new_output(
            output_type="error",
            ename="Error",
            evalue=output.text or "",
            traceback=[],
        )

    if output.data:
        return nbformat.v4.new_output(
            output_type="display_data",
            data=output.data,
            metadata={},
        )

    return nbformat.v4.new_output(output_type="stream", name="stdout", text=output.text or "")
