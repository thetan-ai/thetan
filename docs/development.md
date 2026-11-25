---
title: Development
---

## Build and test

Thetan uses [meson-python](https://meson-python.readthedocs.io) and
[spin](https://github.com/scientific-python/spin). The Cython extensions
are never compiled inside the source tree, so build before importing:

```bash
pip install -r requirements/build.txt -r requirements/test.txt
spin build          # compile into build/ and install into build-install/
spin test                            # run pytest against that build
spin test -t thetan.tests.test_dbscan   # one test module
spin test -- --pyargs thetan -k dbscan  # tests matching a pattern
```

Cython runs with `--warning-errors`, so any Cython warning fails the
build. New Python modules or test files must be listed in
`thetan/meson.build` or `thetan/tests/meson.build` to be installed.

## Build the website

The website is a [MyST](https://mystmd.org) project in `docs/`. mystmd
needs Node.js 18 or later on the `PATH`.

```bash
pip install -r requirements/docs.txt
spin docs                 # build thetan, the API pages, the gallery and the site
spin docs --serve         # same, then serve it on http://127.0.0.1:8000
spin docs --no-plot       # skip running the gallery examples
spin docs --clean         # start from scratch
```

The site lands in `docs/_build/site`: the landing page at its root, the
MyST docs under `docs/`. Pass `--base-url /thetan` when it is served from
a subpath, as on GitHub Pages.

`spin docs` runs two generators before `myst build`:

`tools/docs/build_api.py`
: Imports each module of the built package and writes one page per module
  to `docs/api/`, parsing the docstrings with numpydoc. Add a module to
  its `MODULES` list to document it.

`tools/docs/build_gallery.py`
: Runs every script in `docs/examples/` and writes its page, output and
  figures to `docs/gallery/`. Outputs are cached by the hash of the script,
  so unchanged examples are not run again.

Both output folders are generated; do not edit them by hand.

## Write an example

An example is a Python script in `docs/examples/`, in the sphinx-gallery
layout:

```python
"""
Title of the example
====================

One paragraph that describes it. The first paragraph is also the summary
on the gallery card.
"""

import numpy as np

# %%
# A section title
# ---------------
#
# Comment lines at the top of a cell are rendered as markdown.

x = np.arange(3)
print(x)
```

Cells run in order in one namespace. What they print and the matplotlib
figures they leave open appear under their code, and the first figure is
the thumbnail. Seed the random generators so that the page is the same on
every build.

## Docstrings

Public functions are documented in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.
The API pages render the Parameters, Returns, Raises, Notes and Examples
sections.
