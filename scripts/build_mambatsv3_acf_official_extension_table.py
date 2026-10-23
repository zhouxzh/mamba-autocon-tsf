#!/usr/bin/env python3
"""Render a read-only table of audited official-model adapter extensions."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003")
SOURCES = {
    "patchtst_pems_h192": ("PEMS-BAY", 192, "PatchTST", "official_patchtst_mean_mse", "active_vs_official_patchtst_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h192_extension_20261003/audit.json"),
    "patchtst_pems_h336": ("PEMS-BAY", 336, "PatchTST", "official_patchtst_mean_mse", "active_vs_official_patchtst_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h336_extension_20261003/audit.json"),
    "patchtst_metr_h192": ("METR-LA", 192, "PatchTST", "official_patchtst_mean_mse", "active_vs_official_patchtst_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_metr_la_h192_extension_20261003/audit.json"),
    "patchtst_weather_h192": ("Weather", 192, "PatchTST", "official_patchtst_mean_mse", "active_vs_official_patchtst_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_weather_h192_extension_20261003/audit.json"),
    "itrans_pems_h192": ("PEMS-BAY", 192, "iTransformer", "official_itransformer_mean_mse", "active_vs_official_itransformer_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h192_extension_20261003/audit.json"),
    "itrans_pems_h336": ("PEMS-BAY", 336, "iTransformer", "official_itransformer_mean_mse", "active_vs_official_itransformer_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h336_extension_20261003/audit.json"),
    "itrans_metr_h192": ("METR-LA", 192, "iTransformer", "official_itransformer_mean_mse", "active_vs_official_itransformer_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_metr_la_h192_extension_20261003/audit.json"),
    "timesnet_pems_h192": ("PEMS-BAY", 192, "TimesNet", "official_timesnet_mean_mse", "active_vs_official_timesnet_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h192_extension_20261003/audit.json"),
    "timesnet_pems_h336": ("PEMS-BAY", 336, "TimesNet", "official_timesnet_mean_mse", "active_vs_official_timesnet_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h336_extension_20261003/audit.json"),
    "timemixer_pems_h192": ("PEMS-BAY", 192, "TimeMixer", "official_timemixer_mean_mse", "active_vs_official_timemixer_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h192_extension_20261003/audit.json"),
    "timemixer_pems_h336": ("PEMS-BAY", 336, "TimeMixer", "official_timemixer_mean_mse", "active_vs_official_timemixer_percent", "artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h336_extension_20261003/audit.json"),
}


def main() -> int:
    rows = []
    for _, spec in SOURCES.items():
        target, horizon, model, baseline_key, percent_key, raw_path = spec
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("audit_valid") is not True:
            raise ValueError("invalid external extension audit: {}".format(path))
        summary = payload["summary"]
        rows.append({"target": target, "horizon": horizon, "official_model": model, "active_mse": summary["active_mean_mse"], "official_mse": summary[baseline_key], "active_vs_official_percent": summary[percent_key], "wins": "{}/3".format(summary["active_wins"]), "upper_ratio": summary["active_official_ratio_upper_1se"], "audit": str(path)})
    result = {"scope": "post-release external-model-code extensions; not the frozen primary significance matrix", "rows": rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "official_model_extensions.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Official Model-Code Extensions", "", "Pinned official networks run through local chronological adapters. All rows are post-release external-model-code extensions and remain separate from the frozen primary matrix.", "", "| Target | H | Official model | Active | Official | Active vs official | Wins | One-SE upper ratio |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| {target} | {horizon} | {official_model} | {active_mse:.6f} | {official_mse:.6f} | {active_vs_official_percent:+.3f}% | {wins} | {upper_ratio:.4f} |".format(**row))
    lines.append("")
    lines.append("Official code is adapter-controlled for the audited chronology and validation-first test lifecycle; this is not a bitwise reproduction of each upstream runner's default data lifecycle.")
    (OUTPUT / "official_model_extensions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
