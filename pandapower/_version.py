import importlib.metadata

try:
    __version__ = importlib.metadata.version("pandapower")
except importlib.metadata.PackageNotFoundError:
    # if the package is not installed, try reading the toml itself
    import tomllib
    from pathlib import Path
    toml_file = Path(__file__).parent / "../pyproject.toml"
    if toml_file.exists() and toml_file.is_file():
        with toml_file.open("rb") as f:
            data = tomllib.load(f)
            if "project" in data and "version" in data["project"]:
                __version__ = data["project"]["version"]

__format_version__ = "3.0.0"
