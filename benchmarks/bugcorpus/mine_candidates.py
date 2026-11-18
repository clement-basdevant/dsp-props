"""List the candidate bug-fix commits of the rediscovery corpus (step 1 of the protocol).

Usage: python benchmarks/bugcorpus/mine_candidates.py CLONES [--out candidates.tsv]

CLONES holds full clones of numpy, scipy, librosa and pytorch/audio (``git clone --filter=blob:none
--no-checkout`` is enough). The selection is mechanical, so that the corpus does not depend on
which bugs we remember: every non-merge commit of the window whose message marks a bug fix and
that touches a non-test source file of the modules below. ``docs/rediscovery.md`` describes the
classification that follows (``triage.tsv``).

librosa's commit messages rarely say "fix", so its candidates also include every entry of the
"Bug fixes" sections of its changelog (``docs/changelog.rst``) for the releases of the window
(0.7.2, January 2020, and later); an entry and a commit can describe the same fix.

The window starts on 1 November 2019: Python 3.8 is the oldest interpreter ``uv`` installs, and
the oldest releases with Python 3.8 wheels are NumPy 1.17.3 and SciPy 1.3.2 (both October 2019),
so a bug fixed earlier cannot be replayed from a released wheel.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

SINCE = "2019-11-01"
UNTIL = "2026-11-01"

# (repository directory, upstream commit the history was read at, message pattern, paths)
SOURCES = {
    "numpy": ("numpy", r"^(BUG|FIX)", ["numpy/fft"]),
    "scipy": ("scipy", r"^(BUG|FIX)", ["scipy/signal", "scipy/fft", "scipy/fftpack"]),
    "librosa": ("librosa", r"(?i)\b(fix|fixed|fixes|bug|bugfix)\b",
                ["librosa/core", "librosa/effects.py", "librosa/filters.py"]),
    "torchaudio": ("audio", r"(?i)\b(fix|fixed|fixes|bug|bugfix)\b",
                   ["torchaudio/functional", "src/torchaudio/functional", "torchaudio/transforms",
                    "src/torchaudio/transforms", "torchaudio/csrc/lfilter.cpp",
                    "src/libtorchaudio/lfilter.cpp"]),
}
NOT_SOURCE = re.compile(r"(^|/)tests?/|\.(rst|md|txt|toml|build|pyi|json|msgpack|cfg|ini)$|meson|setup\.py")


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def candidates(clones: pathlib.Path):
    for library, (directory, pattern, paths) in SOURCES.items():
        repo = clones / directory
        head = git(repo, "rev-parse", "HEAD").strip()
        log = git(repo, "log", "--no-merges", f"--since={SINCE}", f"--until={UNTIL}",
                  "--format=%x00%H%x09%ad%x09%s", "--date=short", "--name-only", "--", *paths)
        for entry in log.split("\0")[1:]:
            header, *files = entry.strip("\n").split("\n")
            sha, date, subject = header.split("\t", 2)
            if not re.search(pattern, subject):
                continue
            sources = [f for f in files if f and any(f.startswith(p) for p in paths)
                       and not NOT_SOURCE.search(f)]
            if sources:
                yield library, head, sha[:10], date, subject, " ".join(sources)


LIBROSA_FIRST = (0, 7, 2)


def librosa_changelog(repo: pathlib.Path):
    """(version, entry) for every "Bug fixes" entry of releases >= LIBROSA_FIRST."""
    lines = git(repo, "show", "HEAD:docs/changelog.rst").splitlines()
    version, section, out = None, None, []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        m = re.match(r"^v(\d+)\.(\d+)(?:\.(\d+))?(\.post\d+)?\s*$", line)
        if m and re.match(r"^-{3,}", nxt):
            version = tuple(int(g or 0) for g in m.groups()[:3])
            section = None
            continue
        if line.strip() and not line.startswith(" ") and re.match(r"^[A-Z][A-Za-z ]+$", line.strip()):
            section = line.strip()
            continue
        if section != "Bug fixes" or version is None or version < LIBROSA_FIRST:
            continue
        if line.strip().startswith("- "):
            out.append([".".join(map(str, version)), line.strip()[2:]])
        elif line.startswith("      ") and out and line.strip():
            out[-1][1] += " " + line.strip()
    for version, entry in out:
        refs = re.findall(r"#(\d+)", entry)
        text = re.sub(r"`#\d+`_,?\s*", "", entry).replace("\t", " ")
        yield f"changelog:{'+'.join(refs) or '-'}", version, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clones", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("candidates.tsv"))
    args = parser.parse_args()
    rows = sorted(candidates(args.clones), key=lambda r: (r[0], r[3], r[2]))
    heads = {r[0]: r[1] for r in rows}
    rows += [("librosa", heads["librosa"], ref, f"v{version}", text, "docs/changelog.rst")
             for ref, version, text in librosa_changelog(args.clones / "librosa")]
    lines = [f"# history read at: {', '.join(f'{k} {v[:10]}' for k, v in sorted(heads.items()))}",
             "library\tcommit\tdate\tsubject\tfiles"]
    lines += ["\t".join((lib, sha, date, subject.replace("\t", " "), files))
              for lib, _, sha, date, subject, files in rows]
    args.out.write_text("\n".join(lines) + "\n")
    counts = {lib: sum(r[0] == lib for r in rows) for lib in SOURCES}
    print(f"{len(rows)} candidates: {counts} -> {args.out}")


if __name__ == "__main__":
    main()
