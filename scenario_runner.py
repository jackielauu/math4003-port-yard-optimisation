"""Shared CLI utilities for running scenario JSON files through a MILP model."""

import argparse
import csv
import importlib
import math
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import gurobipy as gp

from solution_payloads import plot_payload_results, write_solution_payload


STATUS_NAMES = {
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.INTERRUPTED: "INTERRUPTED",
}

RESULT_FIELDS = [
    "model",
    "scenario",
    "family",
    "seed",
    "status",
    "objective",
    "runtime_seconds",
    "mip_gap",
    "best_bound",
    "solution_count",
    "num_vars",
    "num_binary_vars",
    "num_constraints",
    "strict_capacity",
    "time_limit_seconds",
    "threads_per_solve",
    "scenario_path",
    "payload_path",
    "plot_path",
    "error",
]


def build_parser(
    description,
    default_scenario_dir,
    default_excel_path,
    default_csv_path,
    default_payload_dir,
    default_plot_dir,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Scenario JSON files or directories. Defaults to --scenario-dir.",
    )
    parser.add_argument(
        "--scenario-dir",
        default=None,
        help=f"Directory containing scenario JSON files. Default: {default_scenario_dir}",
    )
    parser.add_argument(
        "--excel",
        "--xlsx",
        dest="excel_path",
        nargs="?",
        const=default_excel_path,
        default=None,
        help=f"Write results to an Excel .xlsx file. Default path when omitted: {default_excel_path}",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        nargs="?",
        const=default_csv_path,
        default=None,
        help=f"Write results to a CSV file. Default path when omitted: {default_csv_path}",
    )
    parser.add_argument(
        "--payload-dir",
        nargs="?",
        const=default_payload_dir,
        default=None,
        help=(
            "Write detailed per-scenario solution JSON payloads. "
            f"Default path when omitted: {default_payload_dir}"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        nargs="?",
        const=default_plot_dir,
        default=None,
        help=(
            "Write PNG plots from saved solution payloads. "
            f"Default path when omitted: {default_plot_dir}"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use the strict no-overflow capacity variant.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional per-scenario Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of scenario solves to run in parallel. Default: 1.",
    )
    parser.add_argument(
        "--threads-per-solve",
        type=int,
        default=None,
        help="Optional Gurobi Threads value for each scenario solve. Use 1 when running many workers.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser


def scenario_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                for filename in files:
                    if _is_scenario_json(filename):
                        paths.append(os.path.join(root, filename))
        elif os.path.isfile(item) and _is_scenario_json(os.path.basename(item)):
            paths.append(item)
        else:
            raise FileNotFoundError(f"Scenario path does not exist: {item}")
    return sorted(os.path.normpath(path) for path in paths)


def _is_scenario_json(filename):
    lower = filename.lower()
    return lower.endswith(".json") and lower not in {"manifest.json", "benchmark_manifest.json"}


def describe_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))

    family = ""
    seed = ""
    if parent.startswith("seed_"):
        seed = parent.removeprefix("seed_")
        family = grandparent
    else:
        family = parent

    return base, family, seed


def run_one_scenario(
    path,
    model_label,
    load_scenario,
    build_model,
    strict_capacity=False,
    time_limit_seconds=None,
    threads_per_solve=None,
    payload_dir=None,
):
    scenario, family, seed = describe_path(path)
    result = {
        "model": model_label,
        "scenario": scenario,
        "family": family,
        "seed": seed,
        "status": "ERROR",
        "objective": None,
        "runtime_seconds": None,
        "mip_gap": None,
        "best_bound": None,
        "solution_count": None,
        "num_vars": None,
        "num_binary_vars": None,
        "num_constraints": None,
        "strict_capacity": strict_capacity,
        "time_limit_seconds": time_limit_seconds,
        "threads_per_solve": threads_per_solve,
        "scenario_path": path,
        "payload_path": None,
        "plot_path": None,
        "error": None,
    }

    started = time.perf_counter()
    try:
        data = load_scenario(path)
        model, variables = build_model(data, strict_capacity=strict_capacity)
        model.Params.OutputFlag = 0
        if time_limit_seconds is not None:
            model.Params.TimeLimit = time_limit_seconds
        if threads_per_solve is not None:
            model.Params.Threads = threads_per_solve
        model.optimize()

        result["status"] = STATUS_NAMES.get(model.Status, str(model.Status))
        result["runtime_seconds"] = round(_safe_float(getattr(model, "Runtime", None)), 4)
        result["solution_count"] = getattr(model, "SolCount", None)
        result["num_vars"] = getattr(model, "NumVars", None)
        result["num_binary_vars"] = getattr(model, "NumBinVars", None)
        result["num_constraints"] = getattr(model, "NumConstrs", None)

        if model.SolCount > 0:
            result["objective"] = _safe_float(model.ObjVal)
            result["mip_gap"] = _safe_float(_safe_model_attr(model, "MIPGap"))
            result["best_bound"] = _safe_float(_safe_model_attr(model, "ObjBound"))
            if payload_dir:
                result["payload_path"] = write_solution_payload(result, data, variables, payload_dir)
    except gp.GurobiError as exc:
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- report per-scenario load/build errors
        result["error"] = str(exc)
    finally:
        if result["runtime_seconds"] is None:
            result["runtime_seconds"] = round(time.perf_counter() - started, 4)

    return result


def run_scenarios(
    paths,
    model_label,
    load_scenario,
    build_model,
    strict_capacity=False,
    time_limit_seconds=None,
    threads_per_solve=None,
    payload_dir=None,
    workers=1,
    show_progress=False,
):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if threads_per_solve is not None and threads_per_solve < 1:
        raise ValueError("threads_per_solve must be at least 1 when provided")

    if workers == 1 or len(paths) <= 1:
        results = []
        for index, path in enumerate(paths, start=1):
            if show_progress:
                print(f"[{index}/{len(paths)}] {os.path.basename(path)}")
            results.append(
                run_one_scenario(
                    path=path,
                    model_label=model_label,
                    load_scenario=load_scenario,
                    build_model=build_model,
                    strict_capacity=strict_capacity,
                    time_limit_seconds=time_limit_seconds,
                    threads_per_solve=threads_per_solve,
                    payload_dir=payload_dir,
                )
            )
        return results

    max_workers = min(workers, len(paths))
    ordered_results = [None] * len(paths)
    load_module = load_scenario.__module__
    build_module = build_model.__module__

    if show_progress:
        print(f"Running {len(paths)} scenario(s) with {max_workers} worker process(es).")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, path in enumerate(paths):
            payload = {
                "path": path,
                "model_label": model_label,
                "load_module": load_module,
                "load_name": load_scenario.__name__,
                "build_module": build_module,
                "build_name": build_model.__name__,
                "strict_capacity": strict_capacity,
                "time_limit_seconds": time_limit_seconds,
                "threads_per_solve": threads_per_solve,
                "payload_dir": payload_dir,
            }
            futures[executor.submit(_run_one_scenario_from_modules, payload)] = (index, path)

        done = 0
        for future in as_completed(futures):
            index, path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 -- report worker failures as rows
                result = _error_result(
                    path=path,
                    model_label=model_label,
                    strict_capacity=strict_capacity,
                    time_limit_seconds=time_limit_seconds,
                    threads_per_solve=threads_per_solve,
                    error=str(exc),
                )
            ordered_results[index] = result
            done += 1
            if show_progress:
                print(f"[{done}/{len(paths)}] {result['scenario']}: {result['status']}")

    return ordered_results


def _run_one_scenario_from_modules(payload):
    load_module = importlib.import_module(payload["load_module"])
    build_module = importlib.import_module(payload["build_module"])
    load_scenario = getattr(load_module, payload["load_name"])
    build_model = getattr(build_module, payload["build_name"])

    return run_one_scenario(
        path=payload["path"],
        model_label=payload["model_label"],
        load_scenario=load_scenario,
        build_model=build_model,
        strict_capacity=payload["strict_capacity"],
        time_limit_seconds=payload["time_limit_seconds"],
        threads_per_solve=payload["threads_per_solve"],
        payload_dir=payload["payload_dir"],
    )


def _error_result(
    path,
    model_label,
    strict_capacity,
    time_limit_seconds,
    threads_per_solve,
    error,
):
    scenario, family, seed = describe_path(path)
    return {
        "model": model_label,
        "scenario": scenario,
        "family": family,
        "seed": seed,
        "status": "ERROR",
        "objective": None,
        "runtime_seconds": None,
        "mip_gap": None,
        "best_bound": None,
        "solution_count": None,
        "num_vars": None,
        "num_binary_vars": None,
        "num_constraints": None,
        "strict_capacity": strict_capacity,
        "time_limit_seconds": time_limit_seconds,
        "threads_per_solve": threads_per_solve,
        "scenario_path": path,
        "payload_path": None,
        "plot_path": None,
        "error": error,
    }


def run_cli(
    model_label,
    load_scenario,
    build_model,
    default_scenario_dir,
    default_excel_path,
    default_csv_path,
    description,
):
    default_payload_dir = os.path.join("results", "payloads", model_label)
    default_plot_dir = os.path.join("results", "plots", model_label)
    parser = build_parser(
        description,
        default_scenario_dir,
        default_excel_path,
        default_csv_path,
        default_payload_dir,
        default_plot_dir,
    )
    args = parser.parse_args()

    inputs = args.paths or [args.scenario_dir or default_scenario_dir]
    paths = scenario_paths(inputs)
    if args.max_scenarios is not None:
        paths = paths[: args.max_scenarios]
    if not paths:
        raise SystemExit(f"No scenario JSON files found in: {', '.join(inputs)}")

    payload_dir = args.payload_dir
    if args.plot_dir and not payload_dir:
        payload_dir = default_payload_dir

    results = run_scenarios(
        paths=paths,
        model_label=model_label,
        load_scenario=load_scenario,
        build_model=build_model,
        strict_capacity=args.strict,
        time_limit_seconds=args.time_limit,
        threads_per_solve=args.threads_per_solve,
        payload_dir=payload_dir,
        workers=args.workers,
        show_progress=args.workers > 1,
    )

    if args.plot_dir:
        plotted_paths = plot_payload_results(results, args.plot_dir)
        print(f"\nSaved {len(plotted_paths)} PNG plot(s) to {args.plot_dir}")

    print_results(results)

    if payload_dir:
        saved_payloads = sum(1 for result in results if result.get("payload_path"))
        print(f"\nSaved {saved_payloads} solution payload(s) to {payload_dir}")
    if args.excel_path:
        write_xlsx(results, args.excel_path)
        print(f"Saved Excel results to {args.excel_path}")
    if args.csv_path:
        write_csv(results, args.csv_path)
        print(f"Saved CSV results to {args.csv_path}")


def print_results(results):
    header = f"{'scenario':<42} {'status':<12} {'objective':>16} {'runtime':>10}  error"
    print(header)
    print("-" * len(header))
    for result in results:
        scenario = result["scenario"]
        if len(scenario) > 42:
            scenario = scenario[:39] + "..."
        obj = result["objective"]
        obj_str = f"{obj:,.2f}" if isinstance(obj, (int, float)) else "-"
        runtime = result["runtime_seconds"]
        runtime_str = f"{runtime:.2f}s" if isinstance(runtime, (int, float)) else "-"
        print(
            f"{scenario:<42} {result['status']:<12} {obj_str:>16} {runtime_str:>10}  "
            f"{result['error'] or ''}"
        )


def write_csv(results, path):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(_ordered_rows(results))


def write_xlsx(results, path):
    _ensure_parent_dir(path)
    rows = [RESULT_FIELDS] + [[row.get(field) for field in RESULT_FIELDS] for row in _ordered_rows(results)]
    sheet_xml = _worksheet_xml(rows)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", _content_types_xml())
        workbook.writestr("_rels/.rels", _root_rels_xml())
        workbook.writestr("docProps/core.xml", _core_xml(created))
        workbook.writestr("docProps/app.xml", _app_xml())
        workbook.writestr("xl/workbook.xml", _workbook_xml())
        workbook.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        workbook.writestr("xl/styles.xml", _styles_xml())
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _ordered_rows(results):
    return [{field: result.get(field) for field in RESULT_FIELDS} for result in results]


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_model_attr(model, attr_name):
    try:
        return getattr(model, attr_name)
    except gp.GurobiError:
        return None


def _safe_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _worksheet_xml(rows):
    xml_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cells.append(_cell_xml(row_idx, col_idx, value, is_header=row_idx == 1))
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<cols>"
        '<col min="1" max="1" width="20" customWidth="1"/>'
        '<col min="2" max="4" width="30" customWidth="1"/>'
        '<col min="5" max="16" width="18" customWidth="1"/>'
        "</cols>"
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def _cell_xml(row_idx, col_idx, value, is_header=False):
    cell_ref = f"{_column_name(col_idx)}{row_idx}"
    style = ' s="1"' if is_header else ""
    if value is None:
        return f'<c r="{cell_ref}"{style}/>'
    if isinstance(value, bool):
        value = "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{cell_ref}"{style}><v>{value}</v></c>'
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if text.strip() != text else ""
    return f'<c r="{cell_ref}" t="inlineStr"{style}><is><t{preserve}>{text}</t></is></c>'


def _column_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _content_types_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""


def _root_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _workbook_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Results" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def _workbook_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
  </cellXfs>
</styleSheet>"""


def _core_xml(created):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>math4003 scenario runner</dc:creator>
  <cp:lastModifiedBy>math4003 scenario runner</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>"""


def _app_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
</Properties>"""
