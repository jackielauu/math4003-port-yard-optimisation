"""Create PNG plots from saved solution payload JSON files."""

import argparse
import os

from solution_payloads import plot_payload_file


def payload_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                for filename in files:
                    if filename.lower().endswith(".json"):
                        paths.append(os.path.join(root, filename))
        elif os.path.isfile(item) and item.lower().endswith(".json"):
            paths.append(item)
        else:
            raise FileNotFoundError(f"Payload path does not exist: {item}")
    return sorted(os.path.normpath(path) for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "payloads",
        nargs="*",
        help="Solution payload JSON files or directories. Defaults to results/payloads.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots/from_payloads",
        help="Directory where PNG files will be saved.",
    )
    args = parser.parse_args()

    inputs = args.payloads or ["results/payloads"]
    paths = payload_paths(inputs)
    if not paths:
        raise SystemExit(f"No payload JSON files found in: {', '.join(inputs)}")

    saved = []
    for path in paths:
        out_path = plot_payload_file(path, args.output_dir)
        saved.append(out_path)
        print(f"Saved {out_path}")

    print(f"\nSaved {len(saved)} PNG plot(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
