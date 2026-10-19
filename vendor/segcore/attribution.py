"""Read the preserved upstream citation from the project notice file."""
from pathlib import Path


def citation_notice():
    return (Path(__file__).resolve().parents[2] / "NOTICE").read_text(encoding="utf-8")
