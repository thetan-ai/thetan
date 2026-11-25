"""Generate the MyST API reference pages from the numpydoc docstrings.

Run it against a built thetan (``spin docs`` sets PYTHONPATH for this):

    python tools/docs/build_api.py --out docs/api
"""

import argparse
import importlib
import inspect
import re
import shutil
import sys
from pathlib import Path

from numpydoc.docscrape import NumpyDocString

REPO_URL = "https://github.com/thetan-ai/thetan/blob/master"
REPO_ROOT = Path(__file__).resolve().parents[2]

MODULES = (
    {
        "name": "thetan.berserker",
        "title": "Clustering",
        "blurb": "Thetan Berserker (TB), the main clustering method, and TBSCAN, "
                 "which links TB centroids with DBSCAN to find non-convex clusters.",
    },
    {
        "name": "thetan.core",
        "title": "Thetan Sequential",
        "blurb": "Thetan Sequential (TS), the one-pass primitive, and the cleanup "
                 "steps that TB builds from it. Compiled with Cython.",
    },
    {
        "name": "thetan.metrics",
        "title": "Distance metrics",
        "blurb": "Distance functions shared by every Thetan algorithm. Every metric "
                 "is oriented so that a smaller value means two rows are closer. "
                 "Compiled with Cython.",
    },
    {
        "name": "thetan.dbscan",
        "title": "DBSCAN",
        "blurb": "A standalone DBSCAN. Noise points get the label "
                 "`DBSCAN_NOISE` (-1). Compiled with Cython.",
    },
    {
        "name": "thetan.image",
        "title": "Images",
        "blurb": "Helpers that turn images and volumes into feature rows that the "
                 "clustering functions accept, and post-process segmentations. "
                 "Compiled with Cython.",
    },
    {
        "name": "thetan.accuracy",
        "title": "Accuracy",
        "blurb": "Scores that compare a clustering with true labels. Needs scipy "
                 "and scikit-learn.",
    },
    {
        "name": "thetan.patchify",
        "title": "Patches",
        "blurb": "Split images into patches and stitch them back. Needs "
                 "scikit-image.",
    },
    {
        "name": "thetan.viz",
        "title": "Visualization",
        "blurb": "Plotting helpers for clustering results. Needs matplotlib and "
                 "scikit-learn.",
    },
)

PARAM_SECTIONS = (
    "Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Attributes",
)
TEXT_SECTIONS = ("Warnings", "Notes", "References")


def rst_to_md(*, text):
    """Convert the reST markup found in docstrings to MyST markdown.

    Parameters
    ----------
    text : str
        Docstring text in reST.

    Returns
    -------
    str
        The same text in MyST markdown.
    """
    text = re.sub(r":[a-z:]+:`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        match = re.match(r"^(\s*)\.\. (math|note|warning)::\s*(.*)$", lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue
        kind, arg = match.group(2), match.group(3)
        body = [arg] if arg else []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
            body.append(lines[i].strip())
            i += 1
        body = "\n".join(body).strip()
        fence = "```{math}" if kind == "math" else f":::{{{kind}}}"
        close = "```" if kind == "math" else ":::"
        out.extend([fence, body, close, ""])
    return "\n".join(out).strip()


def render_examples(*, lines):
    """Render a numpydoc Examples section, fencing doctest blocks.

    Parameters
    ----------
    lines : list of str
        Raw lines of the section.

    Returns
    -------
    str
        Markdown with each doctest session in a ``pycon`` code block.
    """
    out = []
    block = []
    for line in [*lines, ""]:
        if line.startswith(">>>") or line.startswith("...") or (block and line.strip()):
            block.append(line)
            continue
        if block:
            out.extend(["```pycon", *block, "```", ""])
            block = []
        out.append(line)
    return rst_to_md(text="\n".join(out))


def render_params(*, title, params):
    """Render a numpydoc parameter-like section as a definition list.

    Parameters
    ----------
    title : str
        Section name, for example "Parameters".
    params : list of numpydoc.docscrape.Parameter
        Parsed entries of the section.

    Returns
    -------
    str
        Markdown for the section.
    """
    out = [f"**{title}**", ""]
    for param in params:
        if param.name and param.type:
            term = f"`{param.name}` : *{param.type}*"
        else:
            term = f"*{param.name or param.type}*"
        desc = rst_to_md(text="\n".join(param.desc)) or "No description."
        desc = "\n".join(f"  {line}" if line else "" for line in desc.split("\n"))
        out.extend([term, f": {desc.lstrip()}", ""])
    return "\n".join(out)


def render_doc(*, doc):
    """Render a whole numpydoc docstring as MyST markdown.

    Parameters
    ----------
    doc : str
        Cleaned docstring, as returned by inspect.getdoc.

    Returns
    -------
    str
        Markdown for the summary and every non-empty section.
    """
    if not doc:
        return ":::{note}\nThis function is not documented yet.\n:::"
    parsed = NumpyDocString(doc)
    parts = []
    summary = " ".join(parsed["Summary"]).strip()
    if summary:
        parts.append(rst_to_md(text=summary))
    if parsed["Extended Summary"]:
        parts.append(rst_to_md(text="\n".join(parsed["Extended Summary"])))
    for section in PARAM_SECTIONS:
        if parsed[section]:
            parts.append(render_params(title=section, params=parsed[section]))
    if parsed["See Also"]:
        refs = ", ".join(f"`{name}`" for group in parsed["See Also"]
                         for name, _ in group[0])
        parts.append(f"**See Also**\n\n{refs}")
    for section in TEXT_SECTIONS:
        if parsed[section]:
            body = parsed[section]
            body = body if isinstance(body, list) else [body]
            parts.append(f"**{section}**\n\n" + rst_to_md(text="\n".join(body)))
    if parsed["Examples"]:
        parts.append("**Examples**\n\n" + render_examples(lines=parsed["Examples"]))
    return "\n\n".join(parts)


def public_members(*, module):
    """List the functions and classes a module defines, in source order.

    Parameters
    ----------
    module : module
        Imported module.

    Returns
    -------
    list of tuple
        ``(name, object)`` for each public callable whose ``__module__`` is
        the module itself.
    """
    return [
        (name, obj) for name, obj in vars(module).items()
        if not name.startswith("_") and callable(obj)
        and getattr(obj, "__module__", None) == module.__name__
    ]


def signature(*, name, obj):
    """Format the call signature of a function or class.

    Parameters
    ----------
    name : str
        Public name of the object.
    obj : callable
        Function or class.

    Returns
    -------
    str
        For example ``"berserker(points, theta, metric='euclidean')"``.
    """
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = re.sub(r"\bnumpy\.ndarray\b", "np.ndarray", sig)
    sig = sig.replace("typing.", "")
    prefix = "class " if inspect.isclass(obj) else ""
    return f"{prefix}{name}{sig}"


def aliases(*, obj, package):
    """Find the names under which the top-level package exposes an object.

    Parameters
    ----------
    obj : callable
        Function or class.
    package : module
        The top-level thetan package.

    Returns
    -------
    list of str
        Names in ``package`` bound to ``obj``.
    """
    return [name for name, value in vars(package).items() if value is obj]


def source_link(*, module):
    """Build a link to a module's source file on GitHub.

    Parameters
    ----------
    module : module
        Imported module.

    Returns
    -------
    str or None
        URL of the .py or .pyx file, or None if no source file is found.
    """
    stem = module.__name__.replace(".", "/")
    for suffix in (".pyx", ".py"):
        if (REPO_ROOT / f"{stem}{suffix}").exists():
            return f"{REPO_URL}/{stem}{suffix}"
    return None


def member_label(*, module_name, name):
    """Return the MyST label of an API entry.

    Parameters
    ----------
    module_name : str
        Dotted module name.
    name : str
        Function or class name.

    Returns
    -------
    str
        Label usable as ``[](#label)`` from any page.
    """
    return f"api-{module_name.replace('.', '-')}-{name}".lower()


def module_page(*, spec, module, package):
    """Render the reference page of one module.

    Parameters
    ----------
    spec : dict
        Entry of MODULES.
    module : module
        Imported module.
    package : module
        The top-level thetan package, used to list aliases.

    Returns
    -------
    str
        Markdown for the page.
    """
    out = [
        "---",
        f"title: {spec['title']}",
        f"subtitle: \"{spec['name']}\"",
        "---",
        "",
        spec["blurb"],
        "",
    ]
    link = source_link(module=module)
    if link:
        out.extend([f"Source: [{Path(link).name}]({link})", ""])
    for name, obj in public_members(module=module):
        out.extend([
            f"({member_label(module_name=module.__name__, name=name)})=",
            f"## `{name}`",
            "",
            "```python",
            signature(name=name, obj=obj),
            "```",
            "",
        ])
        names = aliases(obj=obj, package=package)
        if names:
            shown = ", ".join(f"`thetan.{alias}`" for alias in names)
            out.extend([f"Also available as {shown}.", ""])
        out.extend([render_doc(doc=inspect.getdoc(obj) or ""), ""])
    return "\n".join(out)


def index_page(*, entries):
    """Render the API landing page with one summary table per module.

    Parameters
    ----------
    entries : list of tuple
        ``(spec, module)`` for every module that imported.

    Returns
    -------
    str
        Markdown for the page.
    """
    out = [
        "---",
        "title: API reference",
        "---",
        "",
        "Every public function, generated from its numpydoc docstring. The "
        "main entry points are importable from the top-level package: "
        "`from thetan import TB, TS, tbscan, distance_matrix`.",
        "",
    ]
    for spec, module in entries:
        out.extend([
            f"## [{spec['title']}](./{spec['name']}.md)",
            "",
            spec["blurb"],
            "",
            "| Function | Summary |",
            "| --- | --- |",
        ])
        for name, obj in public_members(module=module):
            doc = inspect.getdoc(obj) or ""
            summary = " ".join(NumpyDocString(doc)["Summary"]) if doc else ""
            summary = rst_to_md(text=summary).replace("|", "\\|") or "Not documented yet."
            label = member_label(module_name=module.__name__, name=name)
            out.append(f"| [`{name}`](#{label}) | {summary} |")
        out.append("")
    return "\n".join(out)


def build(*, out_dir):
    """Write the API index and one page per module.

    Parameters
    ----------
    out_dir : Path
        Folder that receives the pages. It is emptied first.

    Returns
    -------
    list of str
        Modules that could not be imported. They get a page that says so,
        so that the table of contents in myst.yml stays valid.
    """
    package = importlib.import_module("thetan")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    entries = []
    skipped = []
    for spec in MODULES:
        try:
            module = importlib.import_module(spec["name"])
        except ImportError as err:
            print(f"Skipping {spec['name']}: {err}", file=sys.stderr)
            skipped.append(spec["name"])
            (out_dir / f"{spec['name']}.md").write_text(
                f"---\ntitle: {spec['title']}\nsubtitle: \"{spec['name']}\"\n---\n\n"
                f"{spec['blurb']}\n\n:::{{warning}}\nThis module could not be "
                f"imported when the docs were built: `{err}`.\n:::\n"
            )
            continue
        entries.append((spec, module))
        page = module_page(spec=spec, module=module, package=package)
        (out_dir / f"{spec['name']}.md").write_text(page)
    (out_dir / "index.md").write_text(index_page(entries=entries))
    return skipped


def main():
    """Parse the command line and build the API pages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for the generated pages.")
    args = parser.parse_args()
    skipped = build(out_dir=args.out)
    print(f"API pages written to {args.out}"
          + (f" (skipped: {', '.join(skipped)})" if skipped else ""))


if __name__ == "__main__":
    main()
