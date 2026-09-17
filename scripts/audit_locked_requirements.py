"""Audit resolved Python versions against OSV without installing deployment wheels."""

import json
from urllib.request import Request, urlopen

from check_locked_requirements import ROOT, pins
from packaging.version import Version


def audit():
    packages = {}
    for filename in ("runtime.lock", "dev.lock"):
        for name, version in pins(ROOT / "requirements" / filename).items():
            packages[name, Version(version).public] = None
    if not packages:
        raise RuntimeError("No locked dependencies found")
    queries = [{"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
               for name, version in packages]
    request = Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": queries}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "BoneMarrowVision-dependency-audit"},
    )
    with urlopen(request, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(packages):
        raise RuntimeError("Incomplete OSV response")
    findings = []
    for (name, version), result in zip(packages, results, strict=True):
        if result.get("next_page_token"):
            raise RuntimeError(f"OSV results are truncated for {name}; inspect the package manually")
        for vulnerability in result.get("vulns", []):
            findings.append(f"{name}=={version}: https://osv.dev/vulnerability/{vulnerability['id']}")
    if findings:
        raise SystemExit("Known dependency advisories:\n" + "\n".join(findings))
    print(f"OSV reported no known advisories for {len(packages)} locked package versions.")


if __name__ == "__main__":
    audit()
