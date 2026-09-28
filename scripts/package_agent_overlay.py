"""Hash-lock a downloaded CPython 3.10/Linux wheelhouse into one persistent file."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def package(wheelhouse: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refuse to overwrite archive")
    rows, requirements = [], []
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise ValueError("wheelhouse empty")
    for wheel in wheels:
        name, version, _, tags = parse_wheel_filename(wheel.name)
        if not any(t.interpreter in {"py2", "py3", "cp310"} and
                   (t.platform == "any" or "manylinux" in t.platform) for t in tags):
            raise ValueError("non CPython-3.10/Linux wheel")
        with zipfile.ZipFile(wheel) as archive:
            metadata_files = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                raise ValueError("ambiguous wheel metadata")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        requires_python = metadata.get("Requires-Python")
        from packaging.specifiers import SpecifierSet
        if requires_python and Version("3.10.4") not in SpecifierSet(requires_python):
            raise ValueError(f"unsupported Python requirement: {name}")
        requires = metadata.get_all("Requires-Dist", [])
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        rows.append({"name": str(name), "version": str(version), "file": wheel.name,
                     "sha256": digest, "bytes": wheel.stat().st_size, "requires": requires})
        requirements.append(f"{name}=={version} --hash=sha256:{digest}")
    installed = {row["name"]: Version(row["version"]) for row in rows}
    environment = {**default_environment(), "python_version": "3.10", "python_full_version": "3.10.4",
                   "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
                   "platform_machine": "x86_64", "extra": ""}
    for row in rows:
        for value in row["requires"]:
            req = Requirement(value)
            if req.marker and not req.marker.evaluate(environment):
                continue
            candidate = installed.get(canonicalize_name(req.name))
            if candidate is None or candidate not in req.specifier:
                raise ValueError(f"missing/incompatible Linux dependency: {row['name']} -> {req}")
    report = {"platform": "CPython 3.10.4 Linux x86_64", "wheel_count": len(rows),
              "total_bytes": sum(r["bytes"] for r in rows), "packages": rows,
              "dependency_metadata_checked": True, "linux_imports_executed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = wheelhouse / "requirements.lock"
    report_path = wheelhouse / "wheelhouse_manifest.json"
    lock.write_bytes(("\n".join(requirements) + "\n").encode())
    report_path.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    with tarfile.open(output, "x:gz") as archive:
        for wheel in wheels:
            archive.add(wheel, arcname="wheels/" + wheel.name)
        archive.add(lock, arcname="requirements.lock")
        archive.add(report_path, arcname="wheelhouse_manifest.json")
    return {"archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size, "wheel_count": len(rows),
            "linux_imports_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.wheelhouse, args.output)))


if __name__ == "__main__":
    main()
