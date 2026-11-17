"""Reusable Exploratory Data Analysis (EDA) script for Synthea CSV and FHIR R4 outputs.

Usage:
    python synthea_eda.py --input-dir path/to/synthea/output/csv --output-dir ./eda_output
    python synthea_eda.py --input-dir path/to/synthea/output/fhir --format fhir --output-dir ./eda_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

CORE_TABLES = [
    "patients",
    "encounters",
    "conditions",
    "medications",
    "procedures",
    "observations",
    "immunizations",
    "allergies",
    "careplans",
    "payers",
    "payer_transitions",
    "providers",
    "organizations",
    "claims",
]


def detect_format(input_dir: Path) -> str:
    """Detect whether input_dir contains Synthea CSV tables or FHIR JSON bundles."""
    if any(input_dir.glob("*.csv")):
        return "csv"
    if any(input_dir.glob("*.json")):
        return "fhir"
    raise FileNotFoundError(
        f"No .csv or .json files found in {input_dir}. Check --input-dir."
    )


def load_csv_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all Synthea CSV files in input_dir with uppercase normalized column names."""
    tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        name = csv_path.stem.lower()
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        tables[name] = df
    return tables


def _extract_coding(resource: dict[str, Any], field: str) -> tuple[str | None, str | None]:
    """Extract first (code, display) pair from a FHIR CodeableConcept field."""
    concept = resource.get(field) or {}
    codings = concept.get("coding") or []
    if codings:
        return codings[0].get("code"), codings[0].get("display") or concept.get("text")
    return None, concept.get("text")


def _ref_id(ref: str | None) -> str | None:
    """Extract resource ID from a FHIR reference string (e.g. 'urn:uuid:...' or 'Patient/123')."""
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref.split("urn:uuid:")[-1]
    return ref.split("/")[-1]


def load_fhir_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse Synthea FHIR R4 JSON Bundles into normalized DataFrames matching CSV columns."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "patients": [],
        "encounters": [],
        "conditions": [],
        "medications": [],
        "procedures": [],
        "observations": [],
        "immunizations": [],
    }

    for json_path in sorted(input_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        if bundle.get("resourceType") != "Bundle":
            continue

        for entry in bundle.get("entry") or []:
            res = entry.get("resource") or {}
            rtype = res.get("resourceType")
            rid = res.get("id")

            if rtype == "Patient":
                exts = {e.get("url", "").split("-")[-1]: e for e in res.get("extension") or []}
                race = _extract_extension_text(exts.get("race"))
                ethnicity = _extract_extension_text(exts.get("ethnicity"))
                buckets["patients"].append(
                    {
                        "ID": rid,
                        "BIRTHDATE": res.get("birthDate"),
                        "DEATHDATE": res.get("deceasedDateTime"),
                        "GENDER": (res.get("gender") or "").upper()[:1] or None,
                        "RACE": race,
                        "ETHNICITY": ethnicity,
                    }
                )
            elif rtype == "Encounter":
                period = res.get("period") or {}
                enc_class = (res.get("class") or {}).get("code")
                code, desc = _extract_coding(res, "type") if isinstance(res.get("type"), dict) else _first_type_coding(res)
                buckets["encounters"].append(
                    {
                        "ID": rid,
                        "START": period.get("start"),
                        "STOP": period.get("end"),
                        "PATIENT": _ref_id((res.get("subject") or {}).get("reference")),
                        "ENCOUNTERCLASS": enc_class,
                        "CODE": code,
                        "DESCRIPTION": desc,
                    }
                )
            elif rtype == "Condition":
                code, desc = _extract_coding(res, "code")
                buckets["conditions"].append(
                    {
                        "START": res.get("onsetDateTime"),
                        "STOP": res.get("abatementDateTime"),
                        "PATIENT": _ref_id((res.get("subject") or {}).get("reference")),
                        "ENCOUNTER": _ref_id((res.get("encounter") or {}).get("reference")),
                        "CODE": code,
                        "DESCRIPTION": desc,
                    }
                )
            elif rtype == "MedicationRequest":
                code, desc = _extract_coding(res, "medicationCodeableConcept")
                buckets["medications"].append(
                    {
                        "START": res.get("authoredOn"),
                        "PATIENT": _ref_id((res.get("subject") or {}).get("reference")),
                        "ENCOUNTER": _ref_id((res.get("encounter") or {}).get("reference")),
                        "CODE": code,
                        "DESCRIPTION": desc,
                    }
                )
            elif rtype == "Procedure":
                code, desc = _extract_coding(res, "code")
                period = res.get("performedPeriod") or {}
                buckets["procedures"].append(
                    {
                        "START": period.get("start") or res.get("performedDateTime"),
                        "STOP": period.get("end"),
                        "PATIENT": _ref_id((res.get("subject") or {}).get("reference")),
                        "ENCOUNTER": _ref_id((res.get("encounter") or {}).get("reference")),
                        "CODE": code,
                        "DESCRIPTION": desc,
                    }
                )
            elif rtype == "Observation":
                code, desc = _extract_coding(res, "code")
                vq = res.get("valueQuantity") or {}
                val = vq.get("value") if vq else res.get("valueString")
                buckets["observations"].append(
                    {
                        "DATE": res.get("effectiveDateTime"),
                        "PATIENT": _ref_id((res.get("subject") or {}).get("reference")),
                        "ENCOUNTER": _ref_id((res.get("encounter") or {}).get("reference")),
                        "CODE": code,
                        "DESCRIPTION": desc,
                        "VALUE": val,
                        "UNITS": vq.get("unit"),
                    }
                )
            elif rtype == "Immunization":
                code, desc = _extract_coding(res, "vaccineCode")
                buckets["immunizations"].append(
                    {
                        "DATE": res.get("occurrenceDateTime"),
                        "PATIENT": _ref_id((res.get("patient") or {}).get("reference")),
                        "ENCOUNTER": _ref_id((res.get("encounter") or {}).get("reference")),
                        "CODE": code,
                        "DESCRIPTION": desc,
                    }
                )

    return {k: pd.DataFrame(v) for k, v in buckets.items() if v}


def _extract_extension_text(ext: dict[str, Any] | None) -> str | None:
    if not ext:
        return None
    for sub in ext.get("extension") or []:
        if sub.get("url") == "text":
            return sub.get("valueString")
    return None


def _first_type_coding(res: dict[str, Any]) -> tuple[str | None, str | None]:
    types = res.get("type") or []
    if types:
        codings = types[0].get("coding") or []
        if codings:
            return codings[0].get("code"), codings[0].get("display") or types[0].get("text")
        return None, types[0].get("text")
    return None, None


def summarize_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return row count, column count, and overall missingness % for each loaded table."""
    rows = []
    for name, df in sorted(tables.items()):
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isna().sum().sum() / total_cells * 100.0) if total_cells else 0.0
        rows.append(
            {
                "table": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_pct": round(missing_pct, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_patient_ages(patients: pd.DataFrame) -> pd.Series:
    """Compute patient age in years at DEATHDATE or dataset max date."""
    birth = pd.to_datetime(patients["BIRTHDATE"], errors="coerce", utc=True).dt.tz_localize(None)
    if "DEATHDATE" in patients.columns:
        death = pd.to_datetime(patients["DEATHDATE"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        death = pd.Series(pd.NaT, index=patients.index)

    ref_date = pd.concat([birth, death]).max()
    if pd.isna(ref_date):
        ref_date = pd.Timestamp.today().normalize()
    end_date = death.fillna(ref_date)
    return ((end_date - birth).dt.days / 365.25).clip(lower=0)


def save_bar_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    horizontal: bool = True,
) -> None:
    """Save a clean bar chart for a value_counts() series."""
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(series) + 1.5)))
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("muted")[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        series.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[0])
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        plt.xticks(rotation=30, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_patients(patients: pd.DataFrame, fig_dir: Path) -> list[str]:
    """Generate demographic statistics and plots."""
    lines = ["## 2. Patient Demographics", ""]
    n_patients = len(patients)
    lines.append(f"- **Total patients**: {n_patients:,}")

    if "BIRTHDATE" in patients.columns:
        ages = compute_patient_ages(patients)
        lines.append(
            f"- **Age (years)**: mean={ages.mean():.1f}, median={ages.median():.1f}, "
            f"min={ages.min():.1f}, max={ages.max():.1f}"
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(ages.dropna(), bins=25, kde=True, ax=ax)
        ax.set_title("Patient Age Distribution")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(fig_dir / "patient_age_distribution.png", dpi=150)
        plt.close(fig)

    if "DEATHDATE" in patients.columns:
        deceased = patients["DEATHDATE"].notna().sum()
        lines.append(f"- **Deceased patients**: {deceased:,} ({deceased / max(n_patients, 1) * 100:.1f}%)")

    for col in ("GENDER", "RACE", "ETHNICITY"):
        if col in patients.columns:
            vc = patients[col].fillna("Unknown").value_counts()
            summary = ", ".join(f"{k}: {v:,} ({v / n_patients * 100:.1f}%)" for k, v in vc.items())
            lines.append(f"- **{col.title()}**: {summary}")
            save_bar_plot(
                vc,
                title=f"Patient {col.title()} Distribution",
                xlabel="Patients",
                ylabel=col.title(),
                out_path=fig_dir / f"patient_{col.lower()}.png",
            )

    for col in ("INCOME", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"):
        if col in patients.columns:
            vals = pd.to_numeric(patients[col], errors="coerce").dropna()
            if not vals.empty:
                lines.append(
                    f"- **{col.replace('_', ' ').title()}**: mean=${vals.mean():,.2f}, "
                    f"median=${vals.median():,.2f}, max=${vals.max():,.2f}"
                )
    lines.append("")
    return lines


def analyze_encounters(encounters: pd.DataFrame, fig_dir: Path, top_n: int) -> list[str]:
    """Generate encounter utilization, temporal, and cost summaries."""
    lines = ["## 3. Encounters & Utilization", ""]
    lines.append(f"- **Total encounters**: {len(encounters):,}")
    if "PATIENT" in encounters.columns:
        per_pt = encounters.groupby("PATIENT").size()
        lines.append(
            f"- **Encounters per patient**: mean={per_pt.mean():.1f}, median={per_pt.median():.1f}, max={per_pt.max():,}"
        )

    if "ENCOUNTERCLASS" in encounters.columns:
        vc = encounters["ENCOUNTERCLASS"].fillna("unknown").value_counts().head(top_n)
        save_bar_plot(
            vc,
            title="Encounters by Class",
            xlabel="Count",
            ylabel="Encounter Class",
            out_path=fig_dir / "encounters_by_class.png",
        )
        lines.append("")
        lines.append("### Encounter Classes")
        lines.append(vc.rename_axis("ENCOUNTERCLASS").to_frame("COUNT").to_markdown())

    if "START" in encounters.columns:
        starts = pd.to_datetime(encounters["START"], errors="coerce", utc=True).dt.tz_localize(None)
        yearly = starts.dt.year.dropna().astype(int).value_counts().sort_index()
        if not yearly.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            yearly.plot(kind="line", marker="o", ax=ax)
            ax.set_title("Encounters by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Encounters")
            fig.tight_layout()
            fig.savefig(fig_dir / "encounters_by_year.png", dpi=150)
            plt.close(fig)

    for col in ("BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE"):
        if col in encounters.columns:
            vals = pd.to_numeric(encounters[col], errors="coerce").dropna()
            if not vals.empty:
                lines.append(
                    f"- **{col.replace('_', ' ').title()}**: mean=${vals.mean():,.2f}, "
                    f"median=${vals.median():,.2f}, total=${vals.sum():,.2f}"
                )
    lines.append("")
    return lines


def analyze_top_codes(
    tables: dict[str, pd.DataFrame], fig_dir: Path, top_n: int
) -> list[str]:
    """Summarize top clinical descriptions across conditions, medications, procedures, allergies, immunizations."""
    lines = ["## 4. Clinical Event Profiles", ""]
    clinical_tables = ["conditions", "medications", "procedures", "allergies", "immunizations"]
    for tname in clinical_tables:
        df = tables.get(tname)
        if df is None or df.empty or "DESCRIPTION" not in df.columns:
            continue
        vc = df["DESCRIPTION"].dropna().value_counts().head(top_n)
        if vc.empty:
            continue
        save_bar_plot(
            vc,
            title=f"Top {len(vc)} {tname.title()}",
            xlabel="Count",
            ylabel="Description",
            out_path=fig_dir / f"top_{tname}.png",
        )
        lines.append(f"### Top {len(vc)} {tname.title()}")
        lines.append(vc.rename_axis("DESCRIPTION").to_frame("COUNT").to_markdown())
        lines.append("")
    return lines


def analyze_observations(observations: pd.DataFrame, fig_dir: Path, top_n: int) -> list[str]:
    """Summarize top observations and numeric distributions for common vitals/labs."""
    lines = ["## 5. Observations (Vitals & Labs)", ""]
    lines.append(f"- **Total observations**: {len(observations):,}")
    if "DESCRIPTION" not in observations.columns:
        return lines

    vc = observations["DESCRIPTION"].dropna().value_counts().head(top_n)
    if not vc.empty:
        lines.append(f"### Top {len(vc)} Observation Types")
        lines.append(vc.rename_axis("DESCRIPTION").to_frame("COUNT").to_markdown())
        lines.append("")

    if "VALUE" in observations.columns:
        numeric_obs = observations.copy()
        numeric_obs["NUM_VALUE"] = pd.to_numeric(numeric_obs["VALUE"], errors="coerce")
        numeric_obs = numeric_obs.dropna(subset=["NUM_VALUE"])
        if not numeric_obs.empty:
            top_numeric = (
                numeric_obs.groupby("DESCRIPTION")["NUM_VALUE"]
                .agg(["count", "mean", "std", "min", "median", "max"])
                .sort_values("count", ascending=False)
                .head(10)
                .round(2)
            )
            lines.append("### Numeric Summary (Top 10 Numeric Observations)")
            lines.append(top_numeric.to_markdown())
            lines.append("")

            # Plot distribution of the most frequent numeric observation (e.g. Body Weight / BMI / BP)
            top_name = top_numeric.index[0]
            subset = numeric_obs.loc[numeric_obs["DESCRIPTION"] == top_name, "NUM_VALUE"]
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(subset, bins=25, kde=True, ax=ax)
            ax.set_title(f"Distribution: {top_name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(fig_dir / "top_numeric_observation.png", dpi=150)
            plt.close(fig)
    return lines


def run_eda(input_dir: Path, output_dir: Path, fmt: str = "auto", top_n: int = 15) -> Path:
    """Execute full Synthea EDA pipeline and write markdown report + figures."""
    resolved_fmt = detect_format(input_dir) if fmt == "auto" else fmt
    if resolved_fmt == "csv":
        tables = load_csv_tables(input_dir)
    elif resolved_fmt == "fhir":
        tables = load_fhir_tables(input_dir)
    else:
        raise ValueError(f"Unsupported format: {resolved_fmt}")

    if not tables:
        raise RuntimeError(f"No Synthea data loaded from {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Synthea Exploratory Data Analysis Report",
        "",
        f"- **Input directory**: `{input_dir}`",
        f"- **Detected format**: `{resolved_fmt}`",
        "",
        "## 1. Dataset Overview",
        "",
        summarize_tables(tables).to_markdown(index=False),
        "",
    ]

    if "patients" in tables and not tables["patients"].empty:
        report_lines.extend(analyze_patients(tables["patients"], fig_dir))
    if "encounters" in tables and not tables["encounters"].empty:
        report_lines.extend(analyze_encounters(tables["encounters"], fig_dir, top_n))

    report_lines.extend(analyze_top_codes(tables, fig_dir, top_n))

    if "observations" in tables and not tables["observations"].empty:
        report_lines.extend(analyze_observations(tables["observations"], fig_dir, top_n))

    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on Synthea CSV or FHIR R4 outputs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with Synthea CSV or FHIR JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("./eda_output"), help="Directory for report and plots.")
    parser.add_argument("--format", choices=["auto", "csv", "fhir"], default="auto", help="Input data format.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top clinical items to show.")
    args = parser.parse_args()

    report_path = run_eda(args.input_dir, args.output_dir, fmt=args.format, top_n=args.top_n)
    print(f"EDA complete. Report written to: {report_path}")


if __name__ == "__main__":
    main()
