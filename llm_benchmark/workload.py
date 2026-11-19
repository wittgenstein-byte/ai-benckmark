"""Workload loaders for Synthea ETL JSONL files and synthetic clinical prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptItem:
    """A single benchmark job payload."""

    job_id: str
    prompt: str
    estimated_tokens: int


_SYNTHETIC_CLINICAL_BLOCK = (
    "Encounter 2024-08-14 (ambulatory): Patient presents for chronic disease follow-up. "
    "Active conditions: Essential hypertension (I10), Type 2 diabetes mellitus without "
    "complications (E11.9), Hyperlipidemia (E78.5), Chronic kidney disease stage 2 (N18.2). "
    "Medications: Lisinopril 20 mg daily, Metformin 1000 mg twice daily, Atorvastatin 40 mg "
    "nightly, Aspirin 81 mg daily. Vitals: BP 134/82 mmHg, HR 74 bpm, BMI 28.4 kg/m2, "
    "Temp 36.7 C, SpO2 98%. Labs: HbA1c 7.1%, eGFR 78 mL/min/1.73m2, LDL 84 mg/dL, "
    "Potassium 4.3 mEq/L, Creatinine 1.02 mg/dL. Plan: Continue current pharmacotherapy, "
    "reinforce dietary sodium restriction, repeat metabolic panel and HbA1c in 3 months.\n"
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _build_synthetic_prompt(job_idx: int, target_tokens: int, unique_prompts: bool) -> PromptItem:
    target_chars = max(128, target_tokens * 4)
    header = (
        f"Patient Record #SYN-{job_idx:05d} | Age: {45 + (job_idx % 40)} | Sex: {'F' if job_idx % 2 == 0 else 'M'}\n"
        if unique_prompts
        else "Patient Record #SYN-SHARED | Age: 62 | Sex: M\n"
    )
    repeats = max(1, (target_chars - len(header)) // len(_SYNTHETIC_CLINICAL_BLOCK) + 1)
    body = (_SYNTHETIC_CLINICAL_BLOCK * repeats)[: max(0, target_chars - len(header))]
    prompt = header + body
    return PromptItem(
        job_id=f"job-{job_idx:05d}",
        prompt=prompt,
        estimated_tokens=_estimate_tokens(prompt),
    )


def _extract_prompt_from_record(record: dict) -> str:
    for key in ("clinical_summary_md", "prompt", "text", "input", "content"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return json.dumps(record, ensure_ascii=False)


def load_workload(
    source: str | None,
    total_jobs: int,
    synthetic_tokens: int = 1500,
    unique_prompts: bool = True,
) -> list[PromptItem]:
    """Load `total_jobs` prompt items from a JSONL file or generate synthetic clinical prompts."""
    if total_jobs <= 0:
        raise ValueError("total_jobs must be > 0")

    if not source or source.lower() == "synthetic":
        return [
            _build_synthetic_prompt(i, synthetic_tokens, unique_prompts)
            for i in range(1, total_jobs + 1)
        ]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt source file not found: {path}")

    base_records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            pid = str(record.get("patient_id") or record.get("id") or f"rec-{idx:05d}")
            prompt_text = _extract_prompt_from_record(record)
            base_records.append((pid, prompt_text))

    if not base_records:
        raise ValueError(f"No valid prompt records found in {path}")

    items: list[PromptItem] = []
    for i in range(total_jobs):
        pid, text = base_records[i % len(base_records)]
        cycle = i // len(base_records)
        if unique_prompts and cycle > 0:
            prompt_text = f"[Benchmark Request #{i + 1:05d} | Source: {pid}]\n{text}"
        elif unique_prompts:
            prompt_text = f"[Benchmark Request #{i + 1:05d}]\n{text}"
        else:
            prompt_text = text

        items.append(
            PromptItem(
                job_id=f"{pid}#{i + 1:05d}",
                prompt=prompt_text,
                estimated_tokens=_estimate_tokens(prompt_text),
            )
        )
    return items
