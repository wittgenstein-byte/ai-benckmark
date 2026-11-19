"""ETL pipeline to group Synthea EHR data by PATIENT_ID into token-efficient clinical summaries.

Supports Synthea CSV directories and FHIR R4 JSON Bundles, outputting compact
per-patient Markdown (.md), structured JSON (.json), and batch JSONL (.jsonl)
ready for GPU LLM inference and benchmarking.

Usage:
    python synthea_etl.py --input-dir path/to/synthea/output/csv --output-dir ./etl_output
    python synthea_etl.py --input-dir path/to/synthea/output/fhir --format fhir --output-dir ./etl_output
    python synthea_etl.py --input-dir ./csv --output-dir ./etl_output --limit 100 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import tiktoken  # type: ignore

    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency
    _TIKTOKEN_ENC = None


CLINICAL_CSV_TABLES = (
    "patients",
    "encounters",
    "conditions",
    "medications",
    "procedures",
    "observations",
    "immunizations",
    "allergies",
    "careplans",
)


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken if available, else ~4 chars/token."""
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return max(1, (len(text) + 3) // 4)


def short_date(value: str | None) -> str:
    """Normalize an ISO timestamp or date string to YYYY-MM-DD."""
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10]


def _parse_date(value: str | None) -> date | None:
    d = short_date(value)
    if len(d) != 10:
        return None
    try:
        return date.fromisoformat(d)
    except ValueError:
        return None


def _compute_age(birthdate: str | None, deathdate: str | None, ref_date: date) -> int | None:
    birth = _parse_date(birthdate)
    if birth is None:
        return None
    end = _parse_date(deathdate) or ref_date
    years = end.year - birth.year - ((end.month, end.day) < (birth.month, birth.day))
    return max(0, years)


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_format(input_dir: Path) -> str:
    """Detect whether input_dir contains Synthea CSV tables or FHIR JSON bundles."""
    if any(input_dir.glob("*.csv")):
        return "csv"
    if any(input_dir.glob("*.json")):
        return "fhir"
    raise FileNotFoundError(
        f"No .csv or .json files found in {input_dir}. Check --input-dir."
    )


def _extract_coding(resource: dict[str, Any], field: str) -> tuple[str, str]:
    concept = resource.get(field) or {}
    codings = concept.get("coding") or []
    if codings:
        code = _clean_str(codings[0].get("code"))
        desc = _clean_str(codings[0].get("display") or concept.get("text"))
        return code, desc
    return "", _clean_str(concept.get("text"))


def _first_type_coding(resource: dict[str, Any]) -> tuple[str, str]:
    types = resource.get("type") or []
    if types:
        codings = types[0].get("coding") or []
        if codings:
            return _clean_str(codings[0].get("code")), _clean_str(
                codings[0].get("display") or types[0].get("text")
            )
        return "", _clean_str(types[0].get("text"))
    return "", ""


def _ref_id(ref: str | None) -> str:
    if not ref:
        return ""
    if ref.startswith("urn:uuid:"):
        return ref.split("urn:uuid:")[-1]
    return ref.split("/")[-1]


def _extract_extension_text(ext: dict[str, Any] | None) -> str:
    if not ext:
        return ""
    for sub in ext.get("extension") or []:
        if sub.get("url") == "text":
            return _clean_str(sub.get("valueString"))
    return ""


def _new_patient_bucket(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "demographics": {},
        "encounters": [],
        "conditions": [],
        "medications": [],
        "procedures": [],
        "observations": [],
        "immunizations": [],
        "allergies": [],
        "careplans": [],
    }


def load_csv_by_patient(
    input_dir: Path,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Stream Synthea CSV tables and group clinical rows by PATIENT_ID.

    Returns (patients_map, total_raw_bytes).
    """
    patients_path = input_dir / "patients.csv"
    if not patients_path.exists():
        # Case-insensitive fallback
        matches = [p for p in input_dir.glob("*.csv") if p.stem.lower() == "patients"]
        if not matches:
            raise FileNotFoundError(f"patients.csv not found in {input_dir}")
        patients_path = matches[0]

    patients: dict[str, dict[str, Any]] = {}
    raw_bytes = 0

    raw_bytes += patients_path.stat().st_size
    with patients_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            norm = {k.strip().upper(): _clean_str(v) for k, v in row.items() if k}
            pid = norm.get("ID", "")
            if not pid:
                continue
            bucket = _new_patient_bucket(pid)
            bucket["demographics"] = {
                "birthdate": short_date(norm.get("BIRTHDATE")),
                "deathdate": short_date(norm.get("DEATHDATE")),
                "gender": norm.get("GENDER", ""),
                "race": norm.get("RACE", ""),
                "ethnicity": norm.get("ETHNICITY", ""),
                "marital": norm.get("MARITAL", ""),
            }
            patients[pid] = bucket
            if limit is not None and len(patients) >= limit:
                break

    if not patients:
        return patients, raw_bytes

    for table_name in CLINICAL_CSV_TABLES:
        if table_name == "patients":
            continue
        csv_path = input_dir / f"{table_name}.csv"
        if not csv_path.exists():
            matches = [p for p in input_dir.glob("*.csv") if p.stem.lower() == table_name]
            if not matches:
                continue
            csv_path = matches[0]

        raw_bytes += csv_path.stat().st_size
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                norm = {k.strip().upper(): _clean_str(v) for k, v in row.items() if k}
                pid = norm.get("PATIENT", "")
                if pid not in patients:
                    continue
                _append_csv_record(patients[pid], table_name, norm)

    return patients, raw_bytes


def _append_csv_record(bucket: dict[str, Any], table_name: str, row: dict[str, str]) -> None:
    if table_name == "encounters":
        bucket["encounters"].append(
            {
                "start": short_date(row.get("START")),
                "stop": short_date(row.get("STOP")),
                "class": row.get("ENCOUNTERCLASS", ""),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "reason": row.get("REASONDESCRIPTION", ""),
            }
        )
    elif table_name == "conditions":
        bucket["conditions"].append(
            {
                "start": short_date(row.get("START")),
                "stop": short_date(row.get("STOP")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
            }
        )
    elif table_name == "medications":
        bucket["medications"].append(
            {
                "start": short_date(row.get("START")),
                "stop": short_date(row.get("STOP")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "reason": row.get("REASONDESCRIPTION", ""),
            }
        )
    elif table_name == "procedures":
        bucket["procedures"].append(
            {
                "date": short_date(row.get("START") or row.get("DATE")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "reason": row.get("REASONDESCRIPTION", ""),
            }
        )
    elif table_name == "observations":
        bucket["observations"].append(
            {
                "date": short_date(row.get("DATE")),
                "category": row.get("CATEGORY", ""),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "value": row.get("VALUE", ""),
                "units": row.get("UNITS", ""),
            }
        )
    elif table_name == "immunizations":
        bucket["immunizations"].append(
            {
                "date": short_date(row.get("DATE")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
            }
        )
    elif table_name == "allergies":
        bucket["allergies"].append(
            {
                "start": short_date(row.get("START")),
                "stop": short_date(row.get("STOP")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "category": row.get("CATEGORY", ""),
                "reaction": row.get("DESCRIPTION1", ""),
                "severity": row.get("SEVERITY1", ""),
            }
        )
    elif table_name == "careplans":
        bucket["careplans"].append(
            {
                "start": short_date(row.get("START")),
                "stop": short_date(row.get("STOP")),
                "code": row.get("CODE", ""),
                "description": row.get("DESCRIPTION", ""),
                "reason": row.get("REASONDESCRIPTION", ""),
            }
        )


def load_fhir_by_patient(
    input_dir: Path,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Load Synthea FHIR R4 JSON Bundles and group resources by PATIENT_ID."""
    patients: dict[str, dict[str, Any]] = {}
    raw_bytes = 0

    for json_path in sorted(input_dir.glob("*.json")):
        if limit is not None and len(patients) >= limit:
            break
        raw_bytes += json_path.stat().st_size
        with json_path.open("r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        if bundle.get("resourceType") != "Bundle":
            continue

        for entry in bundle.get("entry") or []:
            res = entry.get("resource") or {}
            rtype = res.get("resourceType")
            rid = _clean_str(res.get("id"))

            if rtype == "Patient":
                if not rid:
                    continue
                if limit is not None and len(patients) >= limit and rid not in patients:
                    continue
                exts = {
                    e.get("url", "").split("-")[-1]: e
                    for e in (res.get("extension") or [])
                }
                bucket = patients.setdefault(rid, _new_patient_bucket(rid))
                bucket["demographics"] = {
                    "birthdate": short_date(res.get("birthDate")),
                    "deathdate": short_date(res.get("deceasedDateTime")),
                    "gender": _clean_str(res.get("gender")).upper()[:1],
                    "race": _extract_extension_text(exts.get("race")),
                    "ethnicity": _extract_extension_text(exts.get("ethnicity")),
                    "marital": _clean_str(
                        ((res.get("maritalStatus") or {}).get("text"))
                    ),
                }
            elif rtype == "Encounter":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                period = res.get("period") or {}
                code, desc = (
                    _extract_coding(res, "type")
                    if isinstance(res.get("type"), dict)
                    else _first_type_coding(res)
                )
                patients[pid]["encounters"].append(
                    {
                        "start": short_date(period.get("start")),
                        "stop": short_date(period.get("end")),
                        "class": _clean_str((res.get("class") or {}).get("code")),
                        "code": code,
                        "description": desc,
                        "reason": "",
                    }
                )
            elif rtype == "Condition":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "code")
                patients[pid]["conditions"].append(
                    {
                        "start": short_date(res.get("onsetDateTime")),
                        "stop": short_date(res.get("abatementDateTime")),
                        "code": code,
                        "description": desc,
                    }
                )
            elif rtype == "MedicationRequest":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "medicationCodeableConcept")
                patients[pid]["medications"].append(
                    {
                        "start": short_date(res.get("authoredOn")),
                        "stop": "",
                        "code": code,
                        "description": desc,
                        "reason": "",
                    }
                )
            elif rtype == "Procedure":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "code")
                period = res.get("performedPeriod") or {}
                patients[pid]["procedures"].append(
                    {
                        "date": short_date(
                            period.get("start") or res.get("performedDateTime")
                        ),
                        "code": code,
                        "description": desc,
                        "reason": "",
                    }
                )
            elif rtype == "Observation":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "code")
                vq = res.get("valueQuantity") or {}
                val = vq.get("value") if vq else res.get("valueString")
                patients[pid]["observations"].append(
                    {
                        "date": short_date(res.get("effectiveDateTime")),
                        "category": "",
                        "code": code,
                        "description": desc,
                        "value": _clean_str(val),
                        "units": _clean_str(vq.get("unit")),
                    }
                )
            elif rtype == "Immunization":
                pid = _ref_id((res.get("patient") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "vaccineCode")
                patients[pid]["immunizations"].append(
                    {
                        "date": short_date(res.get("occurrenceDateTime")),
                        "code": code,
                        "description": desc,
                    }
                )
            elif rtype == "AllergyIntolerance":
                pid = _ref_id((res.get("patient") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                code, desc = _extract_coding(res, "code")
                patients[pid]["allergies"].append(
                    {
                        "start": short_date(res.get("recordedDate")),
                        "stop": "",
                        "code": code,
                        "description": desc,
                        "category": _clean_str((res.get("category") or [""])[0]),
                        "reaction": "",
                        "severity": "",
                    }
                )
            elif rtype == "CarePlan":
                pid = _ref_id((res.get("subject") or {}).get("reference"))
                if not pid or pid not in patients:
                    continue
                period = res.get("period") or {}
                cats = res.get("category") or []
                code, desc = ("", "")
                if cats and isinstance(cats[0], dict):
                    codings = cats[0].get("coding") or []
                    if codings:
                        code = _clean_str(codings[0].get("code"))
                        desc = _clean_str(
                            codings[0].get("display") or cats[0].get("text")
                        )
                patients[pid]["careplans"].append(
                    {
                        "start": short_date(period.get("start")),
                        "stop": short_date(period.get("end")),
                        "code": code,
                        "description": desc or _clean_str(res.get("title")),
                        "reason": "",
                    }
                )

    return patients, raw_bytes


def _infer_reference_date(patients: dict[str, dict[str, Any]]) -> date:
    """Find latest clinical date across the cohort for deterministic age calculation."""
    latest: date | None = None
    for p in patients.values():
        demo = p["demographics"]
        for key in ("birthdate", "deathdate"):
            d = _parse_date(demo.get(key))
            if d and (latest is None or d > latest):
                latest = d
        for enc in p["encounters"]:
            d = _parse_date(enc.get("start"))
            if d and (latest is None or d > latest):
                latest = d
    return latest or date.today()


def _summarize_conditions(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        desc = rec.get("description") or rec.get("code") or "Unknown"
        code = rec.get("code", "")
        key = (desc, code)
        entry = grouped.setdefault(
            key,
            {
                "description": desc,
                "code": code,
                "onset": rec.get("start", ""),
                "resolved": rec.get("stop", ""),
                "active": not bool(rec.get("stop")),
                "count": 0,
            },
        )
        entry["count"] += 1
        start = rec.get("start", "")
        stop = rec.get("stop", "")
        if start and (not entry["onset"] or start < entry["onset"]):
            entry["onset"] = start
        if not stop:
            entry["active"] = True
            entry["resolved"] = ""
        elif not entry["active"] and (not entry["resolved"] or stop > entry["resolved"]):
            entry["resolved"] = stop

    active = sorted(
        (v for v in grouped.values() if v["active"]),
        key=lambda x: x["onset"],
        reverse=True,
    )
    resolved = sorted(
        (v for v in grouped.values() if not v["active"]),
        key=lambda x: x["resolved"] or x["onset"],
        reverse=True,
    )
    return {"active": active, "resolved": resolved}


def _summarize_medications(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        desc = rec.get("description") or rec.get("code") or "Unknown"
        code = rec.get("code", "")
        key = (desc, code)
        entry = grouped.setdefault(
            key,
            {
                "description": desc,
                "code": code,
                "first_date": rec.get("start", ""),
                "last_date": rec.get("start", ""),
                "stop": rec.get("stop", ""),
                "active": not bool(rec.get("stop")),
                "count": 0,
                "reason": rec.get("reason", ""),
            },
        )
        entry["count"] += 1
        start = rec.get("start", "")
        stop = rec.get("stop", "")
        if start and (not entry["first_date"] or start < entry["first_date"]):
            entry["first_date"] = start
        if start and (not entry["last_date"] or start > entry["last_date"]):
            entry["last_date"] = start
        if not stop:
            entry["active"] = True
            entry["stop"] = ""
        elif not entry["active"] and (not entry["stop"] or stop > entry["stop"]):
            entry["stop"] = stop
        if not entry["reason"] and rec.get("reason"):
            entry["reason"] = rec["reason"]

    active = sorted(
        (v for v in grouped.values() if v["active"]),
        key=lambda x: x["last_date"],
        reverse=True,
    )
    past = sorted(
        (v for v in grouped.values() if not v["active"]),
        key=lambda x: x["stop"] or x["last_date"],
        reverse=True,
    )
    return {"active": active, "past": past}


def _summarize_observations(
    records: list[dict[str, Any]],
    obs_mode: str,
    max_obs: int,
) -> list[dict[str, Any]]:
    if not records:
        return []

    if obs_mode == "latest":
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            desc = rec.get("description") or rec.get("code") or "Unknown"
            code = rec.get("code", "")
            grouped[(desc, code)].append(rec)

        summarized: list[dict[str, Any]] = []
        for (desc, code), items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.get("date", ""))
            latest = items_sorted[-1]
            nums = [
                n for n in (_try_float(i.get("value", "")) for i in items_sorted) if n is not None
            ]
            entry: dict[str, Any] = {
                "description": desc,
                "code": code,
                "date": latest.get("date", ""),
                "value": latest.get("value", ""),
                "units": latest.get("units", ""),
                "count": len(items_sorted),
            }
            if len(nums) >= 2:
                entry["min"] = round(min(nums), 3)
                entry["max"] = round(max(nums), 3)
            summarized.append(entry)

        summarized.sort(key=lambda x: (x["date"], x["description"]), reverse=True)
        return summarized[:max_obs] if max_obs > 0 else summarized

    # "recent" or "all"
    sorted_obs = sorted(records, key=lambda x: x.get("date", ""), reverse=True)
    if obs_mode == "recent" and max_obs > 0:
        sorted_obs = sorted_obs[:max_obs]
    return [
        {
            "description": r.get("description") or r.get("code") or "Unknown",
            "code": r.get("code", ""),
            "date": r.get("date", ""),
            "value": r.get("value", ""),
            "units": r.get("units", ""),
            "count": 1,
        }
        for r in sorted_obs
    ]


def _summarize_simple_events(
    records: list[dict[str, Any]],
    date_key: str,
    max_items: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        desc = rec.get("description") or rec.get("code") or "Unknown"
        code = rec.get("code", "")
        dt = rec.get(date_key, "")
        entry = grouped.setdefault(
            (desc, code),
            {
                "description": desc,
                "code": code,
                "latest_date": dt,
                "count": 0,
            },
        )
        entry["count"] += 1
        if dt and (not entry["latest_date"] or dt > entry["latest_date"]):
            entry["latest_date"] = dt
    items = sorted(grouped.values(), key=lambda x: x["latest_date"], reverse=True)
    return items[:max_items] if max_items > 0 else items


def _summarize_allergies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        desc = rec.get("description") or rec.get("code") or "Unknown"
        if desc not in seen:
            seen[desc] = {
                "description": desc,
                "code": rec.get("code", ""),
                "category": rec.get("category", ""),
                "reaction": rec.get("reaction", ""),
                "severity": rec.get("severity", ""),
                "active": not bool(rec.get("stop")),
            }
    return sorted(seen.values(), key=lambda x: x["description"])


def _summarize_careplans(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        desc = rec.get("description") or rec.get("code") or "Unknown"
        reason = rec.get("reason", "")
        entry = grouped.setdefault(
            (desc, reason),
            {
                "description": desc,
                "reason": reason,
                "start": rec.get("start", ""),
                "stop": rec.get("stop", ""),
                "active": not bool(rec.get("stop")),
            },
        )
        if rec.get("start") and (not entry["start"] or rec["start"] < entry["start"]):
            entry["start"] = rec["start"]
        if not rec.get("stop"):
            entry["active"] = True
            entry["stop"] = ""
    return sorted(
        grouped.values(),
        key=lambda x: (not x["active"], x["start"]),
        reverse=False,
    )


def build_patient_summary(
    bucket: dict[str, Any],
    ref_date: date,
    obs_mode: str = "latest",
    max_obs: int = 40,
    max_encounters: int = 15,
    max_procedures: int = 20,
    include_codes: bool = False,
) -> dict[str, Any]:
    """Build a compact, deduplicated clinical summary dictionary for one patient."""
    demo = bucket["demographics"]
    age = _compute_age(demo.get("birthdate"), demo.get("deathdate"), ref_date)

    encounters_sorted = sorted(
        bucket["encounters"],
        key=lambda x: x.get("start", ""),
        reverse=True,
    )
    enc_class_counts = dict(
        Counter(e.get("class") or "unknown" for e in encounters_sorted).most_common()
    )
    recent_encounters = encounters_sorted[:max_encounters] if max_encounters > 0 else encounters_sorted

    summary: dict[str, Any] = {
        "patient_id": bucket["patient_id"],
        "demographics": {
            "age": age,
            "gender": demo.get("gender", ""),
            "birthdate": demo.get("birthdate", ""),
            "deathdate": demo.get("deathdate", ""),
            "deceased": bool(demo.get("deathdate")),
            "race": demo.get("race", ""),
            "ethnicity": demo.get("ethnicity", ""),
        },
        "allergies": _summarize_allergies(bucket["allergies"]),
        "conditions": _summarize_conditions(bucket["conditions"]),
        "medications": _summarize_medications(bucket["medications"]),
        "observations": _summarize_observations(
            bucket["observations"], obs_mode=obs_mode, max_obs=max_obs
        ),
        "procedures": _summarize_simple_events(
            bucket["procedures"], date_key="date", max_items=max_procedures
        ),
        "immunizations": _summarize_simple_events(
            bucket["immunizations"], date_key="date", max_items=25
        ),
        "careplans": _summarize_careplans(bucket["careplans"]),
        "encounters_summary": {
            "total": len(encounters_sorted),
            "by_class": enc_class_counts,
            "recent": recent_encounters,
        },
        "raw_record_counts": {
            "encounters": len(bucket["encounters"]),
            "conditions": len(bucket["conditions"]),
            "medications": len(bucket["medications"]),
            "observations": len(bucket["observations"]),
            "procedures": len(bucket["procedures"]),
            "immunizations": len(bucket["immunizations"]),
            "allergies": len(bucket["allergies"]),
            "careplans": len(bucket["careplans"]),
        },
        "include_codes": include_codes,
    }
    return summary


def _fmt_code(code: str, include_codes: bool) -> str:
    return f" [{code}]" if include_codes and code else ""


def render_markdown(summary: dict[str, Any], max_tokens: int | None = None) -> tuple[str, bool]:
    """Render a patient summary dict into compact Markdown, optionally trimming to max_tokens.

    Returns (markdown_text, was_truncated).
    """
    include_codes = bool(summary.get("include_codes", False))
    demo = summary["demographics"]

    header_parts = [f"ID: {summary['patient_id']}"]
    if demo.get("age") is not None:
        header_parts.append(f"Age: {demo['age']}")
    if demo.get("gender"):
        header_parts.append(f"Sex: {demo['gender']}")
    if demo.get("birthdate"):
        header_parts.append(f"DOB: {demo['birthdate']}")
    if demo.get("deceased"):
        header_parts.append(f"Deceased: {demo['deathdate']}")
    else:
        header_parts.append("Status: Living")
    race_eth = "/".join(x for x in (demo.get("race"), demo.get("ethnicity")) if x)
    if race_eth:
        header_parts.append(f"Race/Eth: {race_eth}")

    sections: list[tuple[str, list[str]]] = []

    # 1. Allergies
    if summary["allergies"]:
        lines = []
        for a in summary["allergies"]:
            extra = ", ".join(x for x in (a["reaction"], a["severity"]) if x)
            suffix = f" ({extra})" if extra else ""
            lines.append(f"- {a['description']}{_fmt_code(a['code'], include_codes)}{suffix}")
        sections.append(("Allergies", lines))

    # 2. Active & Resolved Conditions
    conds = summary["conditions"]
    if conds["active"] or conds["resolved"]:
        lines = []
        for c in conds["active"]:
            since = f"since {c['onset']}" if c["onset"] else "active"
            lines.append(f"- [Active] {c['description']}{_fmt_code(c['code'], include_codes)} ({since})")
        for c in conds["resolved"][:15]:
            span = f"{c['onset']}..{c['resolved']}".strip(".")
            lines.append(f"- [Resolved] {c['description']}{_fmt_code(c['code'], include_codes)} ({span})")
        sections.append(("Conditions", lines))

    # 3. Medications
    meds = summary["medications"]
    if meds["active"] or meds["past"]:
        lines = []
        for m in meds["active"]:
            cnt = f", x{m['count']}" if m["count"] > 1 else ""
            lines.append(
                f"- [Active] {m['description']}{_fmt_code(m['code'], include_codes)} (started {m['first_date']}{cnt})"
            )
        for m in meds["past"][:15]:
            cnt = f", x{m['count']}" if m["count"] > 1 else ""
            span = f"{m['first_date']}..{m['stop'] or m['last_date']}".strip(".")
            lines.append(
                f"- [Past] {m['description']}{_fmt_code(m['code'], include_codes)} ({span}{cnt})"
            )
        sections.append(("Medications", lines))

    # 4. Observations (Labs / Vitals)
    if summary["observations"]:
        lines = []
        for o in summary["observations"]:
            val_unit = f"{o['value']} {o['units']}".strip()
            rng = ""
            if "min" in o and "max" in o:
                rng = f" [range {o['min']}..{o['max']}, n={o['count']}]"
            elif o["count"] > 1:
                rng = f" [n={o['count']}]"
            dt = f" ({o['date']})" if o["date"] else ""
            lines.append(
                f"- {o['description']}{_fmt_code(o['code'], include_codes)}: {val_unit}{dt}{rng}"
            )
        sections.append(("Labs & Vitals", lines))

    # 5. Procedures
    if summary["procedures"]:
        lines = []
        for p in summary["procedures"]:
            cnt = f", x{p['count']}" if p["count"] > 1 else ""
            lines.append(
                f"- {p['description']}{_fmt_code(p['code'], include_codes)} ({p['latest_date']}{cnt})"
            )
        sections.append(("Procedures", lines))

    # 6. CarePlans
    if summary["careplans"]:
        lines = []
        for cp in summary["careplans"][:10]:
            status = "Active" if cp["active"] else "Completed"
            reason = f" for {cp['reason']}" if cp["reason"] else ""
            lines.append(f"- [{status}] {cp['description']}{reason} ({cp['start']})")
        sections.append(("Care Plans", lines))

    # 7. Immunizations
    if summary["immunizations"]:
        lines = [
            f"- {im['description']} ({im['latest_date']}"
            + (f", x{im['count']})" if im["count"] > 1 else ")")
            for im in summary["immunizations"]
        ]
        sections.append(("Immunizations", lines))

    # 8. Encounters
    enc = summary["encounters_summary"]
    if enc["total"] > 0:
        cls_str = ", ".join(f"{k}: {v}" for k, v in enc["by_class"].items())
        lines = [f"Total: {enc['total']} ({cls_str})"]
        for e in enc["recent"]:
            reason = f" — {e['reason']}" if e["reason"] else ""
            lines.append(f"- {e['start']} | {e['class']} | {e['description']}{reason}")
        sections.append(("Encounters", lines))

    def _assemble(sec_list: list[tuple[str, list[str]]]) -> str:
        out = [f"# Patient Clinical Summary", " | ".join(header_parts), ""]
        for title, sec_lines in sec_list:
            out.append(f"## {title}")
            out.extend(sec_lines)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    text = _assemble(sections)
    if max_tokens is None or estimate_tokens(text) <= max_tokens:
        return text, False

    # Trim lower-priority sections from the tail until within max_tokens
    trimmed_sections = [(title, list(lines)) for title, lines in sections]
    while trimmed_sections and estimate_tokens(text) > max_tokens:
        last_title, last_lines = trimmed_sections[-1]
        if len(last_lines) > 1:
            last_lines.pop()
        else:
            trimmed_sections.pop()
        text = _assemble(trimmed_sections)

    if estimate_tokens(text) > max_tokens:
        char_cap = max(64, max_tokens * 4 - 20)
        text = text[:char_cap].rstrip() + "\n...(truncated)\n"
    else:
        text = text.rstrip() + "\n\n_(trimmed to token budget)_\n"

    return text, True


def run_etl(
    input_dir: Path,
    output_dir: Path,
    data_format: str = "auto",
    out_format: str = "both",
    obs_mode: str = "latest",
    max_obs: int = 40,
    max_encounters: int = 15,
    max_procedures: int = 20,
    max_tokens: int | None = None,
    limit: int | None = None,
    include_codes: bool = False,
) -> dict[str, Any]:
    """Execute the full ETL pipeline and write summaries + manifest + report."""
    fmt = detect_format(input_dir) if data_format == "auto" else data_format
    if fmt == "csv":
        patients, raw_bytes = load_csv_by_patient(input_dir, limit=limit)
    elif fmt == "fhir":
        patients, raw_bytes = load_fhir_by_patient(input_dir, limit=limit)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if not patients:
        raise RuntimeError(f"No patients loaded from {input_dir} (format={fmt}).")

    ref_date = _infer_reference_date(patients)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_md = out_format in ("md", "both")
    write_json = out_format in ("json", "both")
    write_jsonl = out_format in ("jsonl", "both")

    md_dir = output_dir / "md"
    json_dir = output_dir / "json"
    if write_md:
        md_dir.mkdir(parents=True, exist_ok=True)
    if write_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "patients_summary.jsonl"
    jsonl_fh = jsonl_path.open("w", encoding="utf-8") if write_jsonl else None

    token_counts: list[int] = []
    output_bytes = 0
    truncated_count = 0

    try:
        for pid in sorted(patients.keys()):
            summary = build_patient_summary(
                patients[pid],
                ref_date=ref_date,
                obs_mode=obs_mode,
                max_obs=max_obs,
                max_encounters=max_encounters,
                max_procedures=max_procedures,
                include_codes=include_codes,
            )
            md_text, was_truncated = render_markdown(summary, max_tokens=max_tokens)
            if was_truncated:
                truncated_count += 1
            tok = estimate_tokens(md_text)
            token_counts.append(tok)

            summary_out = dict(summary)
            summary_out.pop("include_codes", None)
            summary_out["estimated_tokens"] = tok
            summary_out["truncated"] = was_truncated
            summary_out["clinical_summary_md"] = md_text

            if write_md:
                md_file = md_dir / f"{pid}.md"
                md_file.write_text(md_text, encoding="utf-8")
                output_bytes += md_file.stat().st_size

            if write_json:
                json_file = json_dir / f"{pid}.json"
                payload = json.dumps(summary_out, ensure_ascii=False, indent=2)
                json_file.write_text(payload, encoding="utf-8")
                output_bytes += json_file.stat().st_size

            if jsonl_fh is not None:
                line = json.dumps(summary_out, ensure_ascii=False) + "\n"
                jsonl_fh.write(line)
                output_bytes += len(line.encode("utf-8"))
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()

    sorted_toks = sorted(token_counts)
    p95_idx = min(len(sorted_toks) - 1, int(0.95 * (len(sorted_toks) - 1)))
    compression_ratio = round(raw_bytes / output_bytes, 2) if output_bytes else 0.0

    manifest = {
        "input_dir": str(input_dir),
        "input_format": fmt,
        "output_dir": str(output_dir),
        "out_format": out_format,
        "obs_mode": obs_mode,
        "max_tokens_budget": max_tokens,
        "patients_processed": len(token_counts),
        "truncated_patients": truncated_count,
        "raw_input_bytes": raw_bytes,
        "output_bytes": output_bytes,
        "compression_ratio": compression_ratio,
        "token_estimator": "tiktoken:cl100k_base" if _TIKTOKEN_ENC else "chars_div_4",
        "tokens": {
            "total": sum(token_counts),
            "min": min(token_counts),
            "median": round(statistics.median(token_counts), 1),
            "mean": round(statistics.mean(token_counts), 1),
            "p95": sorted_toks[p95_idx],
            "max": max(token_counts),
        },
    }

    (output_dir / "etl_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Synthea Patient ETL Report",
        "",
        f"- **Input**: `{input_dir}` (`{fmt}`)",
        f"- **Patients processed**: {manifest['patients_processed']:,}",
        f"- **Truncated patients**: {manifest['truncated_patients']:,}",
        f"- **Raw input size**: {raw_bytes / (1024 * 1024):.2f} MB ({raw_bytes:,} bytes)",
        f"- **Output size**: {output_bytes / (1024 * 1024):.2f} MB ({output_bytes:,} bytes)",
        f"- **Compression ratio**: {compression_ratio}x",
        f"- **Token estimator**: `{manifest['token_estimator']}`",
        (
            f"- **Estimated tokens / patient**: "
            f"min={manifest['tokens']['min']}, median={manifest['tokens']['median']}, "
            f"mean={manifest['tokens']['mean']}, p95={manifest['tokens']['p95']}, "
            f"max={manifest['tokens']['max']}"
        ),
        f"- **Total estimated tokens**: {manifest['tokens']['total']:,}",
        "",
    ]
    (output_dir / "etl_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group Synthea EHR data by PATIENT_ID into token-efficient clinical summaries (.md/.json/.jsonl)."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Synthea CSV or FHIR directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for summaries")
    parser.add_argument("--format", choices=("auto", "csv", "fhir"), default="auto", help="Input format")
    parser.add_argument(
        "--out-format",
        choices=("both", "md", "json", "jsonl"),
        default="both",
        help="Output format (both writes md/, json/, and patients_summary.jsonl)",
    )
    parser.add_argument(
        "--obs-mode",
        choices=("latest", "recent", "all"),
        default="latest",
        help="Observation compression mode (default: latest per vital/lab)",
    )
    parser.add_argument("--max-obs", type=int, default=40, help="Max observations per patient (0=unlimited)")
    parser.add_argument("--max-encounters", type=int, default=15, help="Max recent encounters per patient")
    parser.add_argument("--max-procedures", type=int, default=20, help="Max procedure types per patient")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional per-patient token budget for GPU context window safety",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N patients")
    parser.add_argument("--include-codes", action="store_true", help="Include SNOMED/LOINC/RxNorm codes in Markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_etl(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        data_format=args.format,
        out_format=args.out_format,
        obs_mode=args.obs_mode,
        max_obs=args.max_obs,
        max_encounters=args.max_encounters,
        max_procedures=args.max_procedures,
        max_tokens=args.max_tokens,
        limit=args.limit,
        include_codes=args.include_codes,
    )
    print(
        f"ETL complete: {manifest['patients_processed']} patients | "
        f"mean={manifest['tokens']['mean']} tokens/patient | "
        f"max={manifest['tokens']['max']} | "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
