"""Custom spin commands for thetan development."""

import glob
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from functools import partial
from pathlib import Path

import click
from spin.cmds.meson import (
    _get_configured_command,
    _set_pythonpath,
    build_dir_option,
    build_option,
)

DOCS_DIR = Path("docs")
DOCS_BUILD = DOCS_DIR / "_build"
SITE_DIR = DOCS_BUILD / "site"
LANDING_DIR = DOCS_DIR / "_landing"
GENERATED = (DOCS_DIR / "gallery", DOCS_DIR / "sg_execution_times.rst")


def _run(cmd, *, env=None, cwd=None):
    """Run a command, echo it, and exit on failure.

    Parameters
    ----------
    cmd : list of str
        Command and its arguments.
    env : dict, optional
        Environment for the child process.
    cwd : str or Path, optional
        Working directory for the child process.
    """
    click.secho(f"$ {' '.join(str(c) for c in cmd)}", bold=True, fg="bright_blue")
    result = subprocess.run(cmd, env=env, cwd=cwd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _assemble_site(*, html_dir, site_dir, docs_prefix):
    """Put the landing page at the site root and the Sphinx docs under it.

    Parameters
    ----------
    html_dir : Path
        Output of ``sphinx-build -b dirhtml``.
    site_dir : Path
        Final site folder, emptied first.
    docs_prefix : str
        Subfolder of ``site_dir`` that receives the Sphinx docs.
    """
    if site_dir.exists():
        shutil.rmtree(site_dir)
    shutil.copytree(LANDING_DIR, site_dir)
    shutil.copytree(html_dir, site_dir / docs_prefix)
    (site_dir / ".nojekyll").touch()


def _serve(*, site_dir, port, open_browser):
    """Serve the built site over HTTP until interrupted.

    Parameters
    ----------
    site_dir : Path
        Folder to serve.
    port : int
        Local port.
    open_browser : bool
        If True, open the site in the default browser.
    """
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(site_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        click.secho(f"Serving {site_dir} at {url} (Ctrl+C to stop)", fg="bright_green")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


@click.command()
@click.option("--clean", is_flag=True, default=False,
              help="Remove built and generated docs before building.")
@click.option("--plot/--no-plot", "plot", default=True,
              help="Run the gallery examples and capture their figures.")
@click.option("--serve", is_flag=True, default=False,
              help="Serve the built site locally after building.")
@click.option("--port", default=8000, show_default=True, help="Port for --serve.")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the site in a browser (implies --serve).")
@build_option
@build_dir_option
@click.pass_context
def docs(ctx, *, clean, plot, serve, port, open_browser, build, build_dir):
    """📖 Build the website: landing page, guides, API and gallery.

    Sphinx renders the MyST markdown pages, the numpydoc API reference and
    the sphinx-gallery examples, so thetan is built first unless --no-build
    is given. The result is a static site in docs/_build/site, with the
    landing page at its root and the docs in docs/_build/site/docs. All
    links are relative, so the folder can be served from any URL path.

    \f
    Parameters
    ----------
    clean : bool
        Remove docs/_build and the generated gallery before building.
    plot : bool
        Execute the gallery examples. With --no-plot, the gallery shows
        the code only.
    serve : bool
        Serve the site on localhost after building.
    port : int
        Port used by --serve.
    open_browser : bool
        Open the served site in a browser.
    build : bool
        Build thetan before generating the docs.
    build_dir : str
        Meson build directory.
    """
    if clean:
        for path in (DOCS_BUILD, *GENERATED):
            if path.is_dir():
                click.echo(f"Removing {path}")
                shutil.rmtree(path)
            elif path.exists():
                click.echo(f"Removing {path}")
                path.unlink()

    build_cmd = _get_configured_command("build")
    if build and build_cmd:
        click.secho("Invoking `build` prior to building docs:", bold=True,
                    fg="bright_green")
        ctx.invoke(build_cmd, build_dir=build_dir)
    _set_pythonpath(build_dir)

    html_dir = DOCS_BUILD / "html"
    cmd = [sys.executable, "-m", "sphinx", "-b", "dirhtml", "-W", "--keep-going",
           "-j", "auto", ".", str(html_dir.relative_to(DOCS_DIR))]
    if not plot:
        cmd.extend(["-D", "plot_gallery=0"])
    # From the repo root, `python -m` would put the uncompiled thetan/
    # source tree ahead of the built package on sys.path.
    _run(cmd, env=os.environ.copy(), cwd=DOCS_DIR)

    _assemble_site(html_dir=html_dir, site_dir=SITE_DIR, docs_prefix="docs")
    click.secho(f"\nSite built: {SITE_DIR.resolve() / 'index.html'}",
                fg="bright_green")

    if serve or open_browser:
        _serve(site_dir=SITE_DIR, port=port, open_browser=open_browser)


@click.command()
def clean():
    """🧹 Remove build artifacts, built docs and generated docs pages."""
    targets = [
        "build",
        "build-install",
        "dist",
        "docs/_build",
        "docs/gallery",
        "docs/sg_execution_times.rst",
        "**/__pycache__",
        "**/.pytest_cache",
        "**/*.egg-info",
    ]
    for pattern in targets:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                click.echo(f"  rm -r {path}")
                shutil.rmtree(path)
            elif os.path.isfile(path):
                click.echo(f"  rm {path}")
                os.remove(path)
