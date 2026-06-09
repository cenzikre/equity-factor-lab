# nb_code_extract.py

import argparse
import json
from pathlib import Path


def extract_code_from_notebook(notebook_path: Path) -> str:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    if notebook_path.suffix != ".ipynb":
        raise ValueError("Input file must be a .ipynb notebook")

    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_blocks = []

    for i, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])

            if isinstance(source, list):
                source = "".join(source)

            code_blocks.append(
                f"# ===== Code Cell {i} =====\n"
                f"{source.rstrip()}\n"
            )

    return "\n\n".join(code_blocks)


def main():
    parser = argparse.ArgumentParser(
        description="Extract all code cells from a Jupyter notebook."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Notebook file name or path, e.g. notebook1.ipynb",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    notebook_path = Path(args.name)

    if not notebook_path.is_absolute():
        notebook_path = script_dir / notebook_path

    code_text = extract_code_from_notebook(notebook_path)

    output_path = script_dir / f"{notebook_path.stem}_code.txt"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(code_text)

    print(f"Extracted code saved to: {output_path}")


if __name__ == "__main__":
    main()