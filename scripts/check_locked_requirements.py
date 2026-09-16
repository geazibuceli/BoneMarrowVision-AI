"""Fail CI when declared dependencies no longer match the committed lock files."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def declared(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('-r '):
            yield from declared(path.parent / line[3:])
        elif line and not line.startswith(('#', '-')):
            yield Requirement(line)


def pins(path):
    contents = path.read_text(encoding='utf-8')
    result = {canonicalize_name(name): version for name, version in
              re.findall(r'^([\w.-]+)==([^\s;\\]+)', contents, re.MULTILINE)}
    for name, url in re.findall(r'^([\w.-]+) @ (\S+)', contents, re.MULTILINE):
        _, version, _, _ = parse_wheel_filename(Path(unquote(urlparse(url).path)).name)
        result[canonicalize_name(name)] = str(version)
    return result


def main():
    project = tomllib.loads((ROOT / 'ml/pyproject.toml').read_text(encoding='utf-8'))['project']
    runtime = pins(ROOT / 'requirements/runtime.lock')
    development = pins(ROOT / 'requirements/dev.lock')
    for mode, locked in [('runtime', runtime), ('dev', development)]:
        locked_urls = {canonicalize_name(name): url for name, url in re.findall(
            r'^([\w.-]+) @ (\S+)',
            (ROOT / f'requirements/{mode}.lock').read_text(encoding='utf-8'), re.MULTILINE)}
        requirements = [*declared(ROOT / f'requirements/{mode}.in'),
                        *(Requirement(value) for value in project['dependencies'])]
        if mode == 'dev':
            requirements.extend(Requirement(value) for value in project['optional-dependencies']['dev'])
        for requirement in requirements:
            if requirement.marker and not requirement.marker.evaluate():
                continue
            version = locked.get(canonicalize_name(requirement.name))
            if version is None or not requirement.specifier.contains(version, prereleases=True):
                raise SystemExit(f'{mode}.lock does not satisfy {requirement}; regenerate the locks.')
            if requirement.url and locked_urls.get(canonicalize_name(requirement.name)) != requirement.url:
                raise SystemExit(f'{mode}.lock uses a different wheel URL for {requirement.name}.')
    for name, version in runtime.items():
        if development.get(name) != version:
            raise SystemExit(f'Development and runtime locks disagree on {name}.')
    print('Dependency declarations and runtime/development pins agree.')


if __name__ == '__main__':
    main()
