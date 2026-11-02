"""Save and plot detailed optimisation solution payloads."""

import json
import math
import os


SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]

MODE_LABELS = {
    "F": "Full shutdown",
    "P30": "Partial (30%)",
    "P70": "Partial (70%)",
}
MODE_COLORS = dict(zip(MODE_LABELS, CATEGORICAL))


def write_solution_payload(result, data, variables, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    payload = build_solution_payload(result, data, variables)
    path = os.path.join(output_dir, f"{_safe_filename(result['scenario'])}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_solution_payload(result, data, variables):
    return {
        "metadata": {
            "model": result["model"],
            "scenario": result["scenario"],
            "family": result["family"],
            "seed": result["seed"],
            "status": result["status"],
            "objective": result["objective"],
            "mip_gap": result["mip_gap"],
            "best_bound": result["best_bound"],
            "runtime_seconds": result["runtime_seconds"],
            "strict_capacity": result["strict_capacity"],
            "time_limit_seconds": result["time_limit_seconds"],
            "threads_per_solve": result["threads_per_solve"],
            "scenario_path": result["scenario_path"],
        },
        "sets": {
            "days": list(data["D"]),
            "assets": list(data["A"]),
            "zones": list(data["P_"]),
            "products": list(data.get("G", [])),
        },
        "parameters": {
            "zone_min_inventory_tonnes": _json_safe_mapping(data["Z_min"]),
            "zone_max_inventory_tonnes": _json_safe_mapping(data["Z_max"]),
        },
        "solution": {
            "maintenance_schedule": _extract_maintenance_schedule(data, variables),
            "zone_inventory": _extract_zone_inventory(data, variables),
            "product_inventory": _extract_product_inventory(data, variables),
            "throughput": _extract_day_values(data, variables, "mu", "throughput_tonnes"),
            "overflow": _extract_zone_day_values(data, variables, "o", "overflow_tonnes"),
        },
    }


def plot_payload_results(results, output_dir):
    plotted_paths = []
    for result in results:
        payload_path = result.get("payload_path")
        if not payload_path or not os.path.exists(payload_path):
            continue
        plot_path = plot_payload_file(payload_path, output_dir)
        result["plot_path"] = plot_path
        plotted_paths.append(plot_path)
    return plotted_paths


def plot_payload_file(payload_path, output_dir):
    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    os.makedirs(output_dir, exist_ok=True)
    if payload.get("schema_version") == "rule_based_baseline_v1":
        # Instance labels omit the seed; the saved filename uniquely identifies it.
        scenario = os.path.splitext(os.path.basename(payload_path))[0]
    elif payload.get("schema_version") == "rule_based_operational_evaluation_v1":
        scenario = payload["instance"]
    else:
        scenario = payload["metadata"]["scenario"]
    out_path = os.path.join(output_dir, f"{_safe_filename(scenario)}.png")
    plot_payload(payload, out_path)
    return out_path


def plot_payload(payload, out_path):
    if payload.get("schema_version") == "rule_based_baseline_v1":
        return _plot_rule_based_payload(payload, out_path)
    if payload.get("schema_version") == "rule_based_operational_evaluation_v1":
        if not payload.get("daily_throughput_tonnes"):
            schedule_payload = {
                "schedule": payload["maintenance_schedule"],
                "unscheduled_tasks": payload.get("unscheduled_tasks", []),
                "task_count": payload["task_count"],
                "status": payload["operational_status"],
            }
            return _plot_rule_based_payload(schedule_payload, out_path)
        payload = _evaluated_rule_based_plot_payload(payload)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), facecolor=SURFACE)
    ax_gantt, ax_inventory, ax_throughput = axes

    title = payload["metadata"]["scenario"].replace("_", " ").title()
    model = payload["metadata"]["model"].replace("_", " ").title()
    fig.suptitle(f"{title} ({model})", color=INK_PRIMARY, fontsize=13, fontweight="bold")

    _plot_maintenance_gantt(ax_gantt, payload, mpatches)
    _plot_inventory(ax_inventory, payload)
    _plot_throughput(ax_throughput, payload)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _evaluated_rule_based_plot_payload(payload):
    """Adapt saved fixed-schedule operational results to the MILP plot schema."""
    def zone_series(mapping, value_name):
        return {
            zone: [
                {"day": int(day), value_name: value}
                for day, value in sorted(values.items(), key=lambda item: int(item[0]))
            ]
            for zone, values in mapping.items()
        }

    schedule = [
        dict(item, asset=item["asset_id"], operation=operation)
        for operation, item in payload["maintenance_schedule"].items()
    ]
    inventory = zone_series(payload["stockyard_inventory_tonnes"], "inventory_tonnes")
    throughput = [
        {"day": int(day), "throughput_tonnes": value}
        for day, value in sorted(
            payload["daily_throughput_tonnes"].items(), key=lambda item: int(item[0])
        )
    ]
    days = sorted({point["day"] for point in throughput} |
                  {point["day"] for points in inventory.values() for point in points})
    return {
        "metadata": {
            "scenario": payload["instance"],
            "model": (f"Rule based fixed MILP / {payload['commodity_representation']} / "
                      f"{payload['operational_status']}"),
        },
        "sets": {
            "days": days,
            "assets": sorted({item["asset"] for item in schedule}),
            "zones": sorted(inventory),
        },
        "parameters": payload["plot_parameters"],
        "solution": {
            "maintenance_schedule": schedule,
            "zone_inventory": inventory,
            "throughput": throughput,
            "overflow": zone_series(payload.get("stockyard_overflow_tonnes", {}),
                                    "overflow_tonnes"),
        },
    }


def _plot_rule_based_payload(payload, out_path):
    """Plot saved heuristic maintenance decisions without inventing flow results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    schedule = [dict(item, asset=item["asset_id"]) for item in payload["schedule"].values()]
    unscheduled = payload.get("unscheduled_tasks", [])
    assets = sorted({item["asset"] for item in schedule} |
                    {item["asset_id"] for item in unscheduled})
    first_day = min((item["start_day"] for item in schedule), default=1)
    last_day = max((item["start_day"] + item["duration_days"] - 1
                    for item in schedule), default=first_day)
    plotting_payload = {
        "sets": {"assets": assets, "days": [first_day, last_day]},
        "solution": {"maintenance_schedule": schedule},
    }
    fig, ax = plt.subplots(figsize=(12, max(4, len(assets) * 0.28 + 2)),
                           facecolor=SURFACE)
    _plot_maintenance_gantt(ax, plotting_payload, mpatches)
    title = os.path.splitext(os.path.basename(out_path))[0].replace("_", " ")
    fig.suptitle(f"{title}\nRule-based maintenance — {payload['status']}",
                 color=INK_PRIMARY, fontsize=12)
    note = f"Scheduled: {len(schedule)} / {payload['task_count']}."
    if unscheduled:
        note += " Unscheduled tasks: " + ", ".join(str(item["task_id"]) for item in unscheduled)
    else:
        note += " Unscheduled tasks: 0."
    import textwrap
    note = "\n".join(textwrap.wrap(note, width=120))
    note += "\nInventory and throughput are not included in this schedule payload."
    footer_height = 0.2 + 0.16 * len(note.splitlines())
    fig.set_figheight(fig.get_figheight() + footer_height)
    fig.text(0.02, 0.02, note, fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, footer_height / fig.get_figheight(), 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _extract_maintenance_schedule(data, variables):
    op_asset = {op: asset for asset, ops in data["M_a"].items() for op in ops}
    durations = data["p"]
    schedule = []
    for (operation, mode, start_day), var in variables["x"].items():
        if _var_value(var) > 0.5:
            duration = durations[(operation, mode)]
            schedule.append(
                {
                    "asset": op_asset.get(operation, ""),
                    "operation": operation,
                    "mode": mode,
                    "start_day": start_day,
                    "duration_days": duration,
                    "end_day_exclusive": start_day + duration,
                }
            )
    return sorted(schedule, key=lambda item: (item["asset"], item["start_day"], item["operation"]))


def _extract_zone_inventory(data, variables):
    zone_inventory = {}
    if data.get("is_multicommodity"):
        z = variables["z"]
        for zone in data["P_"]:
            zone_inventory[zone] = [
                {
                    "day": day,
                    "inventory_tonnes": _clean_number(sum(_var_value(z[zone, product, day]) for product in data["PRODUCTS"])),
                }
                for day in data["D"]
            ]
    else:
        z = variables["z"]
        for zone in data["P_"]:
            zone_inventory[zone] = [
                {"day": day, "inventory_tonnes": _clean_number(_var_value(z[zone, day]))}
                for day in data["D"]
            ]
    return zone_inventory


def _extract_product_inventory(data, variables):
    if not data.get("is_multicommodity"):
        return {}

    z = variables["z"]
    product_inventory = {}
    for zone in data["P_"]:
        product_inventory[zone] = {}
        for product in data["PRODUCTS"]:
            product_inventory[zone][product] = [
                {"day": day, "inventory_tonnes": _clean_number(_var_value(z[zone, product, day]))}
                for day in data["D"]
            ]
    return product_inventory


def _extract_day_values(data, variables, variable_name, value_name):
    variable = variables[variable_name]
    return [
        {"day": day, value_name: _clean_number(_var_value(variable[day]))}
        for day in data["D"]
    ]


def _extract_zone_day_values(data, variables, variable_name, value_name):
    variable = variables[variable_name]
    return {
        zone: [
            {"day": day, value_name: _clean_number(_var_value(variable[zone, day]))}
            for day in data["D"]
        ]
        for zone in data["P_"]
    }


def _plot_maintenance_gantt(ax, payload, mpatches):
    assets = sorted(payload["sets"]["assets"])
    schedule = payload["solution"]["maintenance_schedule"]
    y_pos = {asset: i for i, asset in enumerate(assets)}

    for item in schedule:
        mode = item["mode"]
        ax.broken_barh(
            [(item["start_day"], item["duration_days"])],
            (y_pos[item["asset"]] - 0.35, 0.7),
            facecolors=MODE_COLORS.get(mode, CATEGORICAL[0]),
            edgecolors=SURFACE,
            linewidth=1,
        )

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(assets, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(-0.5, max(len(assets) - 0.5, 0.5))
    ax.invert_yaxis()
    _set_day_axis(ax, payload)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Maintenance schedule", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, keep_spines={"bottom"})

    handles = [
        mpatches.Patch(color=color, label=MODE_LABELS[mode])
        for mode, color in MODE_COLORS.items()
        if any(item["mode"] == mode for item in schedule)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=3)


def _plot_inventory(ax, payload):
    inventory = payload["solution"]["zone_inventory"]
    zones = sorted(inventory)
    colors = dict(zip(zones, CATEGORICAL))
    values = [point["inventory_tonnes"] for points in inventory.values() for point in points]
    observed_max = max(values) if values else 0.0
    ylim_top = observed_max * 1.15 if observed_max > 0 else 1.0

    zone_max = payload["parameters"]["zone_max_inventory_tonnes"]
    relevant_max = [zone_max[zone] for zone in zones if zone_max.get(zone, 0) <= max(observed_max * 1.5, 1.0)]
    if relevant_max:
        ylim_top = max(ylim_top, max(relevant_max) * 1.02)

    for zone in zones:
        days = [point["day"] for point in inventory[zone]]
        tonnes = [point["inventory_tonnes"] for point in inventory[zone]]
        ax.plot(days, tonnes, color=colors[zone], linewidth=2, label=zone)
        if zone_max.get(zone, 0) <= ylim_top:
            ax.axhline(zone_max[zone], color=colors[zone], linewidth=0.8, linestyle="--", alpha=0.5)

    _set_day_axis(ax, payload)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Tonnes", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Stockyard inventory (dashed = zone max)", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, keep_spines={"bottom", "left"})
    if zones:
        ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=len(zones))


def _plot_throughput(ax, payload):
    throughput = payload["solution"]["throughput"]
    days = [point["day"] for point in throughput]
    tonnes = [point["throughput_tonnes"] for point in throughput]
    ax.plot(days, tonnes, color=CATEGORICAL[2], linewidth=2)
    _set_day_axis(ax, payload)
    top = max(tonnes) * 1.15 if tonnes and max(tonnes) > 0 else 1.0
    ax.set_ylim(0, top)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Tonnes", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Daily throughput", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, keep_spines={"bottom", "left"})


def _set_day_axis(ax, payload):
    days = payload["sets"]["days"]
    if days:
        ax.set_xlim(days[0], days[-1] + 1)


def _style_axes(ax, keep_spines):
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep_spines)
        if side in keep_spines:
            spine.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def _json_safe_mapping(mapping):
    return {str(key): _clean_number(value) for key, value in mapping.items()}


def _var_value(var):
    return float(var.X)


def _clean_number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return round(number, 6)
    return value


def _safe_filename(name):
    chars = []
    for char in str(name):
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("_")
    filename = "".join(chars).strip("._")
    return filename or "scenario"
