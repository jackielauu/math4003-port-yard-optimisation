import json
from collections import Counter
from pathlib import Path

from PIL import Image
from solution_payloads import _evaluated_rule_based_plot_payload

for rep in ("multi", "single"):
    root = Path("results") / f"rule_based_baseline_evaluated_{rep}_v3"
    out = Path("results/plots") / root.name
    paths = list(root.rglob("*.json"))
    statuses = Counter()
    for path in paths:
        payload = json.loads(path.read_text())
        statuses[payload["operational_status"]] += 1
        with Image.open(out / f"{payload['instance']}.png") as image:
            image.verify()
        if payload["daily_throughput_tonnes"]:
            adapted = _evaluated_rule_based_plot_payload(payload)
            assert len(adapted["solution"]["maintenance_schedule"]) == payload["scheduled_count"]
            assert sum(x["throughput_tonnes"] for x in adapted["solution"]["throughput"]) == sum(payload["daily_throughput_tonnes"].values())
            for zone, points in adapted["solution"]["zone_inventory"].items():
                assert {str(p["day"]): p["inventory_tonnes"] for p in points} == payload["stockyard_inventory_tonnes"][zone]
    assert len(list(out.glob("*.png"))) == len(paths)
    print(rep, len(paths), "verified PNGs", dict(statuses))
