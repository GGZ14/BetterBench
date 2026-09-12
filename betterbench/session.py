"""The interactive compare session: a stdlib, GET-only, stateless HTTP
server over a runs directory.

`make_server` binds `127.0.0.1` on port 0 (the kernel picks a free
port; the caller reads it back from `server.server_address[1]`), so
two sessions started at once can never collide on a fixed port. The
handler serves three routes and re-scans the runs directory on every
request — `list_runs` is called fresh each time, so a run that lands
while the session is open shows up on the next click; there is no
cache anywhere in this module. Slugs are only routed when they equal
a directory name in the current listing *and* survive the shape check
(no `/`, no `..` segment, no leading `.` after decoding), so a
percent-encoded traversal can never resolve to a real path. Unknown
paths, absent slugs and skipped runs each get a distinct plain-text
404. `start` is the top-level entry the CLI's no-args `compare`
branch calls.
"""
from __future__ import annotations

import html
import json
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .gallery import list_runs, render_gallery, render_pair_page
from .html_report import render_html

_THE_404_PLAIN = "text/plain; charset=utf-8"
_WRAP_TARGET = '<div class="wrap">\n'


def _shape_ok(slug: str) -> bool:
    """The decoded-slug shape check: no `/`, no `..` segment, no
    leading `.`. A failing slug can never be turned back into a path
    under the runs dir — including a percent-encoded traversal that
    only *decodes* into one."""
    if not slug or slug.startswith("."):
        return False
    return "/" not in slug and ".." not in slug.split("/")


def _run_js(slug: str) -> str:
    """The 15-line change-navigation script for the run page's
    session bar — the pair page's toolbar (Task 3's `_PAIR_JS`) copied
    for a single select and a fixed A, with no shared constant:
    changing the `Compare…` select navigates to
    `/pair?a=<current>&b=<selection>`."""
    cur = json.dumps(urllib.parse.quote(str(slug), safe=""))
    return (
        "/* The in-situ compare bar: changing the select navigates to\n"
        "   GET /pair?a=<current slug>&b=<selC value> via a plain relative-path\n"
        "   assignment — it works on any port; no URL construction needed.\n"
        "   The select carries every other reportable run's URL-encoded slug;\n"
        "   the swap is a plain anchor to the swapped pair, so it needs no JS. */\n"
        "var selC = document.getElementById('sel-c');\n"
        f"var curSlug = {cur};\n"
        "function pairNav() {\n"
        "  /* Keep A first, B second, exactly as the page header reads them. */\n"
        "  location = \"/pair?a=\" + curSlug + \"&b=\" + selC.value;\n"
        "}\n"
        "selC.addEventListener('change', pairNav);\n"
        "/* (handled by the single select: A is always the current run.\n"
        "   A missing selection can't happen; the select always defaults to the\n"
        "   first other option and is never cleared.) */\n")


def _session_bar(slug: str, reportable: list) -> str:
    """The bar injected above a single run's report header: a
    `← All runs` anchor back at the gallery plus a `Compare…`
    `select` whose options are every *other* reportable slug,
    URL-encoded as the pair page encodes its selects' values."""
    opts = []
    for e in reportable:
        if e.slug == slug:
            continue
        val = html.escape(urllib.parse.quote(str(e.slug), safe=""))
        opts.append(f'<option value="{val}">{html.escape(str(e.slug))}</option>')
    return ('<div class="cmt"><a href="/">← All runs</a>'
            '<span>Compare…</span>'
            '<select id="sel-c">\n' + "\n".join(opts) + '\n</select></div>')


class _Handler(BaseHTTPRequestHandler):
    """One GET-only handler per session server. `runs_dir` is injected
    by `make_server` (see there) into a per-server subclass, so this
    class stays stateless; every request re-scans the directory — a
    fresh `list_runs` each time, no cache.

    404 bodies (plain text), in exactly these three forms: an unknown
    path → `not found`; a valid-shape slug that isn't in the listing →
    `not found — no such run: <slug>`; a listed-but-skipped slug
    (corrupt or ab-only dir) → `not found — <RunEntry.error>`.
    """

    def log_message(self, fmt, *args):
        """No-op: suppresses the per-request access-log spam. To debug,
        drop this override (or temporarily have it call
        `super().log_message(fmt, *args)`) so the base class method
        prints again."""
        pass

    def do_GET(self):
        sp = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(sp.path)
        reportable, skipped = list_runs(self.runs_dir)
        if path == "/":
            self._send(200, render_gallery(self.runs_dir))
            return
        if len(path) > len("/run/") and path.startswith("/run/"):
            self._run_page(path[len("/run/"):], reportable, skipped)
            return
        if path == "/pair":
            # `parse_qs` already unquotes each value exactly once — the
            # wire carries the once-encoded slugs, so a second
            # `urllib.parse.unquote` here would corrupt a name that
            # contains a literal `%` (e.g. `a%2520b` → `a b`).
            qs = {k: v[0]
                 for k, v in urllib.parse.parse_qs(sp.query).items() if v}
            for slug in (qs.get("a"), qs.get("b")):
                bad = self._slug_404(slug, reportable, skipped)
                if bad:
                    self._send(404, bad, _THE_404_PLAIN)
                    return
            self._send(200, render_pair_page(self.runs_dir,
                                            qs.get("a"), qs.get("b")))
            return
        self._send(404, "not found\n", _THE_404_PLAIN)

    def _run_page(self, slug: str, reportable: list, skipped: list):
        bad = self._slug_404(slug, reportable, skipped)
        if bad:
            self._send(404, bad, _THE_404_PLAIN)
            return
        entry = next(e for e in reportable if e.slug == slug)
        page = render_html(entry.results)
        bar = _session_bar(slug, reportable)
        if _WRAP_TARGET in page:
            # Inject the session bar above the report header; when the
            # target were missing (impossible with the current template,
            # guarded anyway) the unmodified page is served.
            page = page.replace(_WRAP_TARGET,
                               _WRAP_TARGET + bar + "\n", 1)
        else:
            page = render_html(entry.results)
        if "</body>" in page:
            page = page.replace("</body>",
                               "<script>\n" + _run_js(slug)
                               + "</script>\n</body>", 1)
        self._send(200, page)

    def _slug_404(self, slug, reportable: list, skipped: list):
        """The 404 body for `slug`, or `None` when it is reportable."""
        if slug is not None and _shape_ok(slug) \
                and any(e.slug == slug for e in reportable):
            return None
        if slug is not None and slug in {e.slug for e in skipped}:
            e = next(e for e in skipped if e.slug == slug)
            return f"not found — {e.error}\n"
        return f"not found — no such run: {slug}\n"

    def _send(self, status: int, body: str,
              ctype: str = "text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(runs_dir: Path, open_browser: bool = False) -> ThreadingHTTPServer:
    """Build a session server for `runs_dir`, always bound to
    `127.0.0.1` on port 0 — the kernel allocates the free port and
    the caller reads it back from `server_address[1]`, so two
    sessions that start at once can never share a port.

    `runs_dir` is injected via a handler factory: `type` builds a
    per-server subclass `type("H", (_Handler,), {"runs_dir": runs_dir})`
    and the factory passes that class to the server, so the module
    defines one stateless handler and each server carries its own
    directory. The server is returned *already serving* —
    `serve_forever` runs on a daemon thread — and is closed with
    `shutdown()` then `server_close()`. When `open_browser` is true,
    `webbrowser.open` is called on the gallery URL best-effort — any
    exception is swallowed, the URL is printed by `start` either way.
    """
    handler = type("H", (_Handler,), {"runs_dir": runs_dir})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    if open_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{srv.server_address[1]}/")
        except Exception:
            pass
    return srv


def start(runs_dir: Path) -> None:
    """The top-level entry the CLI's no-args `compare` calls. Exits
    with a `no reportable runs` message when the dir is missing or
    holds no reportable entries (a dir full of corrupt/ab-only entries
    still *has* runs, so the message says *reportable*); otherwise
    prints the run count, the directory and the URL with
    `Ctrl-C to stop`, and waits until Ctrl-C (the serve loop is the
    daemon thread `make_server` already started; on exit, `shutdown()`
    stops the loop and then `server_close()` cleans up)."""
    entries, _ = list_runs(runs_dir)
    if not entries:
        sys.exit(f"no reportable runs in {runs_dir} — run 'betterbench run' first")
    srv = make_server(runs_dir, open_browser=True)
    print(f"runs: {len(entries)}")
    print(f"serving {runs_dir}")
    print(f"open: http://127.0.0.1:{srv.server_address[1]}/ — Ctrl-C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.server_close()
