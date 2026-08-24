#!/usr/bin/env python3
"""Compatibility wrapper for single-commodity benchmark generation.

The implementation lives in generate_dissertation_scenarios.py. This wrapper
exists only for old scripts/notebooks that still call the former filename.
"""

from __future__ import annotations

from pathlib import Path

from generate_dissertation_scenarios import (
    GeneratorConfig,
    generate_benchmark_suite,
    parse_seeds,
)


def main() -> int:
    output_dir = Path("singlecommodity_scenarios")
    generated = generate_benchmark_suite(
        output_dir,
        config=GeneratorConfig(commodity_count=1),
        seeds=parse_seeds("1-10"),
    )
    print(f"Generated {len(generated) - 1} single-commodity instance file(s) in {output_dir}")
    print(f"Manifest: {generated['benchmark_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
