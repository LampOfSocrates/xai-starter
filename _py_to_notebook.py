"""Convert lesson scripts (*.py) into interactive teaching notebooks (*.ipynb).

For each script we produce a .ipynb where:
  - the module docstring becomes the intro markdown,
  - imports + module-level constants become a "Setup" code cell,
  - every top-level function/class becomes a markdown cell (its docstring,
    rendered as explanation) followed by a clean code cell (docstring stripped),
  - the `if __name__ == "__main__"` guard becomes a final "Run it" cell that
    executes the lesson step by step instead of calling main().

Usage:
  python _py_to_notebook.py <path-or-glob> [<path-or-glob> ...]
  python _py_to_notebook.py            # default: gnn/gnn_l*.py + plm/plm_*.py

Files whose name starts with '_' are skipped (helper scripts like this one).
"""

import ast
import nbformat as nbf


def rst_to_md(text):
    """Light conversion of the scripts' RST-ish docstrings to markdown.

    A line underlined by all '=' becomes an H2; all '-' becomes an H3.
    Everything else is passed through verbatim (it already reads well).
    """
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        u = nxt.strip()
        if line.strip() and u and set(u) == {"="} and len(u) >= 3:
            out.append(f"## {line.strip()}")
            i += 2
            continue
        if line.strip() and u and set(u) == {"-"} and len(u) >= 3:
            out.append(f"### {line.strip()}")
            i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip()


def src_without_docstring(node, source_lines):
    """Return the source of a def/class node with its leading docstring removed."""
    start = node.lineno - 1
    end = node.end_lineno
    block = source_lines[start:end]

    body0 = node.body[0]
    is_doc = (
        isinstance(body0, ast.Expr)
        and isinstance(getattr(body0, "value", None), ast.Constant)
        and isinstance(body0.value.value, str)
    )
    if is_doc:
        # Drop the docstring's physical lines (relative to the node start).
        d_lo = body0.lineno - 1 - start
        d_hi = body0.end_lineno - start
        block = block[:d_lo] + block[d_hi:]
    return "\n".join(block).rstrip("\n"), (ast.get_docstring(node) or "")


def build_notebook(py_path, ipynb_path):
    source = open(py_path, encoding="utf-8").read()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    nb = nbf.v4.new_notebook()
    cells = []

    # --- intro from module docstring ---
    mod_doc = ast.get_docstring(tree) or ""
    cells.append(nbf.v4.new_markdown_cell(rst_to_md(mod_doc)))

    # --- run-order note (notebooks build on each other cell-by-cell) ---
    cells.append(nbf.v4.new_markdown_cell(
        "> **Run order matters.** The cells below build on each other. Run them "
        "**top to bottom** (Jupyter: *Run → Run All Cells*; VS Code: *Run All*). "
        "If you hit `NameError: name 'torch' is not defined` (or similar), you "
        "skipped the **Setup** cell — run it first."
    ))

    # --- walk top-level nodes ---
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr):  # module docstring node
        body = body[1:]

    setup_buf = []  # consecutive imports / simple constant assignments
    setup_done = [False]  # header emitted only before the FIRST setup cell

    def flush_setup():
        if setup_buf:
            code = "\n".join(setup_buf).strip()
            if code:
                if not setup_done[0]:
                    cells.append(nbf.v4.new_markdown_cell(
                        "## Setup — imports & configuration\n\n"
                        "**Run this cell first.** It imports every library and "
                        "defines the module-level constants the rest of the "
                        "notebook relies on."
                    ))
                    setup_done[0] = True
                cells.append(nbf.v4.new_code_cell(code))
            setup_buf.clear()

    name_kind = {ast.Import: "imp", ast.ImportFrom: "imp",
                 ast.Assign: "assign", ast.AnnAssign: "assign"}

    for node in body:
        if type(node) in name_kind:
            seg = ast.get_source_segment(source, node)
            setup_buf.append(seg)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush_setup()
            code, doc = src_without_docstring(node, source_lines)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            heading = f"### `{node.name}` ({kind})"
            md = heading + ("\n\n" + rst_to_md(doc) if doc.strip() else "")
            cells.append(nbf.v4.new_markdown_cell(md))
            cells.append(nbf.v4.new_code_cell(code))
            continue

        if isinstance(node, ast.If):
            # The `if __name__ == "__main__": main()` guard. Replace with an
            # explicit call so the notebook runs top-to-bottom.
            flush_setup()
            cells.append(nbf.v4.new_markdown_cell(
                "## Run the lesson\n\nExecute everything above, then run `main()`."
            ))
            cells.append(nbf.v4.new_code_cell("main()"))
            continue

        # Anything else (stray top-level statement) -> code cell.
        flush_setup()
        seg = ast.get_source_segment(source, node)
        if seg:
            cells.append(nbf.v4.new_code_cell(seg))

    flush_setup()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {ipynb_path}  ({len(cells)} cells)")


if __name__ == "__main__":
    import glob
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    patterns = sys.argv[1:] or [
        os.path.join(here, "gnn", "gnn_l*.py"),
        os.path.join(here, "plm", "plm_*.py"),
    ]

    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    seen = set()
    for py in sorted(paths):
        base = os.path.basename(py)
        if base.startswith("_") or base in seen:
            continue
        seen.add(base)
        ipynb = py[:-3] + ".ipynb"
        build_notebook(py, ipynb)
