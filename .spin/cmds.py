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
GENERATED_DIRS = (DOCS_DIR / "api", DOCS_DIR / "gallery")
SITE_URL_CONFIG = DOCS_BUILD / "site-url.yml"


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


def _write_site_url(*, site_url):
    """Write the MyST config fragment that links the logo to the landing page.

    Parameters
    ----------
    site_url : str
        Absolute URL of the landing page.
    """
    SITE_URL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SITE_URL_CONFIG.write_text(
        "version: 1\nsite:\n  options:\n"
        f"    logo_url: \"{site_url}\"\n"
    )


def _assemble_site(*, html_dir, site_dir, docs_prefix):
    """Put the landing page at the site root and the MyST docs under it.

    Parameters
    ----------
    html_dir : Path
        Output of ``myst build --html``.
    site_dir : Path
        Final site folder, emptied first.
    docs_prefix : str
        Subfolder of ``site_dir`` that receives the MyST docs.
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
@click.option("--base-url", default="", metavar="URL",
              help="Path the site is served under, e.g. /thetan for GitHub Pages.")
@click.option("--site-url", default=None, metavar="URL",
              help="Absolute URL of the landing page, used by the docs logo. "
                   "Defaults to the local --serve address.")
@click.option("--serve", is_flag=True, default=False,
              help="Serve the built site locally after building.")
@click.option("--port", default=8000, show_default=True, help="Port for --serve.")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the site in a browser (implies --serve).")
@build_option
@build_dir_option
@click.pass_context
def docs(ctx, *, clean, plot, base_url, site_url, serve, port, open_browser,
         build, build_dir):
    """📖 Build the MyST website: landing page, guides, API and gallery.

    The API pages and the example gallery are generated from the built
    package, so thetan is built first unless --no-build is given. The
    result is a static site in docs/_build/site, with the landing page at
    its root and the MyST docs in docs/_build/site/docs.

    \f
    Parameters
    ----------
    clean : bool
        Remove docs/_build, docs/api and docs/gallery before building.
    plot : bool
        Execute the gallery examples. With --no-plot, the gallery shows
        the code only.
    base_url : str
        URL path the site is deployed under, without a trailing slash.
    site_url : str or None
        Absolute URL of the landing page. None means the address that
        --serve uses.
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
    if shutil.which("myst") is None:
        click.secho("`myst` not found. Install the docs requirements first:\n"
                    "  pip install -r requirements/docs.txt", fg="bright_red")
        sys.exit(1)

    if clean:
        for path in (DOCS_BUILD, *GENERATED_DIRS):
            if path.exists():
                click.echo(f"Removing {path}")
                shutil.rmtree(path)

    build_cmd = _get_configured_command("build")
    if build and build_cmd:
        click.secho("Invoking `build` prior to building docs:", bold=True,
                    fg="bright_green")
        ctx.invoke(build_cmd, build_dir=build_dir)
    _set_pythonpath(build_dir)
    env = os.environ.copy()

    _run([sys.executable, "tools/docs/build_api.py", "--out", "docs/api"], env=env)
    gallery_cmd = [sys.executable, "tools/docs/build_gallery.py",
                   "--examples", "docs/examples", "--out", "docs/gallery"]
    if not plot:
        gallery_cmd.append("--no-plot")
    _run(gallery_cmd, env=env)

    base_url = base_url.rstrip("/")
    if site_url is None:
        site_url = f"http://127.0.0.1:{port}{base_url}/"
    _write_site_url(site_url=site_url)

    docs_prefix = "docs"
    env["BASE_URL"] = f"{base_url}/{docs_prefix}"
    _run(["myst", "build", "--html", "--strict"], env=env, cwd=DOCS_DIR)

    _assemble_site(html_dir=DOCS_BUILD / "html", site_dir=SITE_DIR,
                   docs_prefix=docs_prefix)
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
        "docs/api",
        "docs/gallery",
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
