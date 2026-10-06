"""Package only reviewed website assets and aggregate evidence for static hosting."""

import argparse
import shutil
from pathlib import Path

from gridpfn.paths import ROOT
from gridpfn.release_evidence import load_evidence
from scripts.carbon_estimate import load_carbon

SITE_FILES = (
    "index.html",
    "overview.css",
    "performance.html",
    "performance.css",
    "performance.js",
    "performance.json",
    "carbon.json",
    "performance.svg",
    "performance.pdf",
    "foundation-comparison.svg",
    "process.svg",
)
MEDIA_FILES = ("gridpfn-training.png", "gridpfn-assistant.mp4", "gridpfn-assistant.gif")


def build(destination):
    destination = Path(destination).resolve()
    if destination == ROOT or destination == ROOT / "site":
        raise ValueError("Choose a separate output directory")
    load_evidence(ROOT / "site/performance.json")
    load_carbon(ROOT / "site/carbon.json", ROOT / "site/performance.json")
    # Enumerate exactly what may be published; never copy the source checkout.
    sources = [(ROOT / "site" / name, destination / name) for name in SITE_FILES]
    sources += [
        (ROOT / "docs/assets" / name, destination / "assets" / name) for name in MEDIA_FILES
    ]
    sources.extend((ROOT / name, destination / name) for name in ("LICENSE", "NOTICE"))
    for source, target in sources:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing regular site asset: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / ".nojekyll").write_text("")
    (destination / "README.md").write_text(
        "# GridPFN website\n\nStatic project overview and interactive simulation results.\n"
        "Built from [Nousphera/GridPFN](https://github.com/Nousphera/GridPFN).\n\n"
        "This site contains aggregate experimental evidence and a generated-household walkthrough. "
        "It contains no household input traces, model weights or credentials.\n\n"
        "Serve locally with `python -m http.server 8767`.\n"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/site-public")
    print(build(parser.parse_args().output))
