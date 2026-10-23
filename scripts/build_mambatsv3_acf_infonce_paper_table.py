#!/usr/bin/env python3
"""Render the audited ACF-InfoNCE release table without model selection."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003")
SOURCES = {
    "PEMS-BAY": Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/audit.json"),
    "METR-LA": Path("artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/audit.json"),
    "Weather": Path("artifacts/paper_final/mambatsv3_acf_infonce_weather_h192_confirmatory_20261003/audit.json"),
}


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_valid") is not True or payload.get("test_metrics_produced") is not True:
        raise ValueError("invalid audited release: {}".format(path))
    return payload


def table_row(target, horizon, summary):
    return {
        "target": target, "horizon": horizon,
        "active": summary["active_mean_mse"],
        "nlinear": summary["nlinear_mean_mse"],
        "patchtst": summary["patchtst_mean_mse"],
        "itransformer": summary["itransformer_mean_mse"],
        "vs_nlinear": summary["active_vs_nlinear_percent"],
        "vs_patchtst": summary["active_vs_patchtst_percent"],
        "vs_itransformer": summary["active_vs_itransformer_percent"],
        "wins": "3/3",
    }


def main() -> int:
    pems = load(SOURCES["PEMS-BAY"])["summary"]
    metr = load(SOURCES["METR-LA"])["summary"]
    weather = load(SOURCES["Weather"])["summary"]
    rows = [
        table_row("PEMS-BAY", 192, pems["H192"]),
        table_row("PEMS-BAY", 336, pems["H336"]),
        table_row("METR-LA", 192, metr),
        table_row("Weather", 192, weather),
    ]
    payload = {
        "rows": rows,
        "source_audits": {name: str(path) for name, path in SOURCES.items()},
        "frozen_primary_baselines": ["NLinear", "PatchTST", "iTransformer"],
        "scope": "Three-seed audited held-out results; lower MSE is better.",
        "post_release_excluded": ["PEMS-BAY TimesNet", "PEMS-BAY TimeMixer", "PEMS-BAY RLinear"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "paper_table.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Audited Main Table",
        "",
        "Three paired seeds, frozen validation selection before test reading, and lower MSE is better. PEMS-BAY post-release TimesNet/TimeMixer/RLinear extensions are intentionally excluded from this primary table.",
        "",
        "| Target | H | Active | NLinear | PatchTST | iTransformer | vs NLinear | vs PatchTST | vs iTransformer | Wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {target} | {horizon} | {active:.6f} | {nlinear:.6f} | {patchtst:.6f} | {itransformer:.6f} | {vs_nlinear:+.3f}% | {vs_patchtst:+.3f}% | {vs_itransformer:+.3f}% | {wins} |".format(**row)
        )
    lines += [
        "",
        "Weather and METR-LA frozen releases additionally include TimesNet. Refer to their individual audit readouts for that baseline and the closed-validation stress tables.",
    ]
    (OUTPUT / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
