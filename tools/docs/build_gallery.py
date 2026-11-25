"""Turn the example scripts into MyST gallery pages.

Each script in the examples folder follows the sphinx-gallery layout: a
module docstring whose first line is the title and whose following
paragraphs describe the example, then cells separated by ``# %%`` lines.
The comment lines at the top of a cell are rendered as markdown, and the
rest of the cell as code. Cells run in order in one namespace; their
printed output and their matplotlib figures are added under the code.

    python tools/docs/build_gallery.py --examples docs/examples --out docs/gallery
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

CACHE_NAME = ".cache.json"
FIGURE_DPI = 110


def parse_example(*, source):
    """Split an example script into its title, description and cells.

    Parameters
    ----------
    source : str
        Content of the script.

    Returns
    -------
    title : str
        First line of the module docstring.
    description : str
        Rest of the module docstring, as markdown.
    cells : list of tuple
        ``(text, code)`` for each cell, both as str.
    """
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    doc_lines = docstring.strip().split("\n")
    title = doc_lines[0].strip() if doc_lines and doc_lines[0].strip() else "Untitled"
    rest = doc_lines[1:]
    if rest and set(rest[0].strip()) <= set("=-~") and rest[0].strip():
        rest = rest[1:]
    description = "\n".join(rest).strip()

    body = source.split("\n")
    if tree.body and isinstance(tree.body[0], ast.Expr) and docstring:
        body = body[tree.body[0].end_lineno:]

    chunks = [[]]
    for line in body:
        if line.strip().startswith("# %%"):
            chunks.append([])
        else:
            chunks[-1].append(line)

    cells = []
    for chunk in chunks:
        text = []
        i = 0
        while i < len(chunk) and (chunk[i].startswith("#") or (not chunk[i].strip() and not text)):
            if chunk[i].startswith("#"):
                text.append(chunk[i][2:] if chunk[i].startswith("# ") else chunk[i][1:])
            i += 1
        code = "\n".join(chunk[i:]).strip("\n")
        text = "\n".join(text).strip()
        if text or code.strip():
            cells.append((text, code))
    return title, description, cells


def run_cells(*, cells, path, image_dir, stem):
    """Execute the cells of one example and collect their outputs.

    Parameters
    ----------
    cells : list of tuple
        ``(text, code)`` pairs from parse_example.
    path : Path
        Path of the script, used for tracebacks and ``__file__``.
    image_dir : Path
        Folder that receives the figures.
    stem : str
        Prefix of the figure file names.

    Returns
    -------
    outputs : list of tuple
        ``(stdout, figure_names)`` for each cell.
    seconds : float
        Total execution time.
    """
    namespace = {"__name__": "__main__", "__file__": str(path)}
    outputs = []
    n_figures = 0
    start = time.perf_counter()
    plt.close("all")
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
    for _, code in cells:
        buffer = io.StringIO()
        if code.strip():
            with contextlib.redirect_stdout(buffer):
                exec(compile(code, str(path), "exec"), namespace)
        names = []
        for number in plt.get_fignums():
            n_figures += 1
            name = f"{stem}_{n_figures:02d}.png"
            plt.figure(number).savefig(image_dir / name, dpi=FIGURE_DPI,
                                       bbox_inches="tight")
            names.append(name)
        plt.close("all")
        outputs.append((buffer.getvalue(), names))
    return outputs, time.perf_counter() - start


def render_page(*, stem, title, description, cells, outputs, seconds):
    """Render the MyST page of one example.

    Parameters
    ----------
    stem : str
        File name of the script without its suffix.
    title : str
        Page title.
    description : str
        Markdown shown below the title.
    cells : list of tuple
        ``(text, code)`` pairs.
    outputs : list of tuple or None
        ``(stdout, figure_names)`` per cell, or None when the examples
        were not run.
    seconds : float or None
        Execution time, or None when the examples were not run.

    Returns
    -------
    str
        Markdown for the page.
    """
    thumbnail = first_figure(outputs=outputs)
    out = ["---", f"title: {json.dumps(title)}"]
    if thumbnail:
        out.append(f"thumbnail: images/{thumbnail}")
    out.extend(["---", "", description, "",
                f":::{{tip}} Download the source: {{download}}`{stem}.py <./{stem}.py>`",
                ":::", ""])
    for index, (text, code) in enumerate(cells):
        if text:
            out.extend([text, ""])
        if code.strip():
            out.extend(["```python", code, "```", ""])
        if outputs is None:
            continue
        stdout, figures = outputs[index]
        if stdout.strip():
            out.extend(["```text", stdout.rstrip(), "```", ""])
        for name in figures:
            out.extend([f"![Figure produced by the cell above](images/{name})", ""])
    if seconds is not None:
        out.append(f"*Running time: {seconds:.1f} s.*")
    return "\n".join(out) + "\n"


def first_figure(*, outputs):
    """Return the name of the first figure an example produced.

    Parameters
    ----------
    outputs : list of tuple or None
        ``(stdout, figure_names)`` per cell.

    Returns
    -------
    str or None
        File name of the first figure, or None.
    """
    for _, names in outputs or []:
        if names:
            return names[0]
    return None


def render_index(*, entries):
    """Render the gallery index as a grid of cards.

    Parameters
    ----------
    entries : list of dict
        One dict per example with the keys stem, title, summary and
        thumbnail.

    Returns
    -------
    str
        Markdown for the page.
    """
    out = [
        "---",
        "title: Gallery",
        "---",
        "",
        "Worked examples, run when the site is built. Each page shows the "
        "code, what it prints and the figures it draws, and links to the "
        "script so you can run it yourself.",
        "",
        "::::{grid} 1 2 2 3",
        "",
    ]
    for entry in entries:
        out.extend([f":::{{card}} {entry['title']}", f":link: ./{entry['stem']}.md", ""])
        if entry["thumbnail"]:
            out.extend([f"![]({'images/' + entry['thumbnail']})", ""])
        out.extend([entry["summary"], ":::", ""])
    out.append("::::")
    return "\n".join(out) + "\n"


def summary_of(*, description):
    """Return the first sentence of a description, on one line.

    Parameters
    ----------
    description : str
        Markdown description of an example.

    Returns
    -------
    str
        First sentence of the first paragraph, with its line breaks
        removed.
    """
    paragraph = description.strip().split("\n\n")[0]
    paragraph = " ".join(line.strip() for line in paragraph.split("\n"))
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph, maxsplit=1)[0]


def build_example(*, path, out_dir, image_dir, plot, cache):
    """Render one example, reusing the cached page when it is unchanged.

    Parameters
    ----------
    path : Path
        Example script.
    out_dir : Path
        Gallery folder.
    image_dir : Path
        Folder for the figures.
    plot : bool
        Run the example and capture its outputs.
    cache : dict
        Maps a script name to the hash and index entry of its last run.
        Updated in place.

    Returns
    -------
    dict
        Index entry with the keys stem, title, summary and thumbnail.
    """
    source = path.read_text()
    stem = path.stem
    digest = hashlib.sha256(source.encode()).hexdigest()
    page = out_dir / f"{stem}.md"
    shutil.copy2(path, out_dir / path.name)
    cached = cache.get(stem)
    if plot and cached and cached["hash"] == digest and cached["plot"] and page.exists():
        print(f"  {path.name}: unchanged, reusing outputs")
        return cached["entry"]

    title, description, cells = parse_example(source=source)
    outputs, seconds = None, None
    if plot:
        for old in image_dir.glob(f"{stem}_*.png"):
            old.unlink()
        print(f"  {path.name}: running")
        outputs, seconds = run_cells(cells=cells, path=path, image_dir=image_dir,
                                     stem=stem)
    page.write_text(render_page(stem=stem, title=title, description=description,
                                cells=cells, outputs=outputs, seconds=seconds))
    entry = {
        "stem": stem,
        "title": title,
        "summary": summary_of(description=description),
        "thumbnail": first_figure(outputs=outputs),
    }
    cache[stem] = {"hash": digest, "plot": plot, "entry": entry}
    return entry


def build(*, examples_dir, out_dir, plot):
    """Render every example and the gallery index.

    Parameters
    ----------
    examples_dir : Path
        Folder with the example scripts.
    out_dir : Path
        Gallery folder.
    plot : bool
        Run the examples and capture their outputs.

    Returns
    -------
    list of str
        Names of the examples that failed to run.
    """
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / CACHE_NAME
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    scripts = sorted(examples_dir.glob("*.py"))
    stems = {path.stem for path in scripts}
    for stale in out_dir.glob("*.md"):
        if stale.stem not in stems and stale.name != "index.md":
            stale.unlink()

    entries = []
    failed = []
    for path in scripts:
        try:
            entries.append(build_example(path=path, out_dir=out_dir,
                                         image_dir=image_dir, plot=plot,
                                         cache=cache))
        except Exception:
            traceback.print_exc()
            failed.append(path.name)
            cache.pop(path.stem, None)
    cache_path.write_text(json.dumps(cache, indent=2))
    (out_dir / "index.md").write_text(render_index(entries=entries))
    return failed


def main():
    """Parse the command line and build the gallery."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True,
                        help="Folder with the example scripts.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for the gallery pages.")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Do not run the examples; show their code only.")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    print(f"Building gallery from {args.examples}")
    failed = build(examples_dir=args.examples, out_dir=args.out, plot=args.plot)
    if failed:
        print(f"Examples failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print(f"Gallery written to {args.out}")


if __name__ == "__main__":
    main()
