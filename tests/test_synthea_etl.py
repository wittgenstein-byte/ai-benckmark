"""Unit and integration tests for synthea_etl.py."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from synthea_etl import run_etl


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        writer.writerows(rows)


class TestSyntheaETL(unittest.TestCase):
    def test_csv_etl_grouping_dedup_and_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "csv"
            out_dir = tmp_path / "out"
            in_dir.mkdir()

            _write_csv(
                in_dir / "patients.csv",
                ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "SSN"],
                [
                    ["p-1", "1970-05-10", "", "M", "asian", "nonhispanic", "999-00-1111"],
                    ["p-2", "1955-01-01", "2024-02-01", "F", "white", "nonhispanic", "999-00-2222"],
                ],
            )
            _write_csv(
                in_dir / "encounters.csv",
                ["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE", "DESCRIPTION", "REASONDESCRIPTION"],
                [
                    ["e-1", "2024-01-10T09:00:00Z", "2024-01-10T09:30:00Z", "p-1", "ambulatory", "185349003", "Encounter for check up", ""],
                    ["e-2", "2024-06-15T11:00:00Z", "2024-06-15T11:30:00Z", "p-1", "emergency", "50849002", "Emergency room admission", "Chest pain"],
                ],
            )
            _write_csv(
                in_dir / "conditions.csv",
                ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
                [
                    ["2020-03-01", "", "p-1", "e-1", "38341003", "Hypertension"],
                    ["2021-01-01", "2021-01-14", "p-1", "e-1", "444814009", "Viral sinusitis"],
                ],
            )
            _write_csv(
                in_dir / "medications.csv",
                ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONDESCRIPTION"],
                [
                    ["2020-03-01", "", "p-1", "e-1", "314076", "Lisinopril 10 MG", "Hypertension"],
                    ["2021-03-01", "", "p-1", "e-1", "314076", "Lisinopril 10 MG", "Hypertension"],
                    ["2022-03-01", "", "p-1", "e-1", "314076", "Lisinopril 10 MG", "Hypertension"],
                ],
            )
            _write_csv(
                in_dir / "observations.csv",
                ["DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
                [
                    ["2023-01-10", "p-1", "e-1", "vital-signs", "8480-6", "Systolic Blood Pressure", "138", "mm[Hg]"],
                    ["2024-01-10", "p-1", "e-1", "vital-signs", "8480-6", "Systolic Blood Pressure", "128", "mm[Hg]"],
                    ["2024-06-15", "p-1", "e-2", "vital-signs", "8480-6", "Systolic Blood Pressure", "145", "mm[Hg]"],
                ],
            )
            _write_csv(
                in_dir / "allergies.csv",
                ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "CATEGORY", "DESCRIPTION1", "SEVERITY1"],
                [["2018-05-01", "", "p-1", "e-1", "91936005", "Allergy to penicillin", "medication", "Hives", "MODERATE"]],
            )

            manifest = run_etl(
                input_dir=in_dir,
                output_dir=out_dir,
                data_format="auto",
                out_format="both",
                obs_mode="latest",
            )

            self.assertEqual(manifest["patients_processed"], 2)
            self.assertEqual(manifest["input_format"], "csv")
            self.assertTrue((out_dir / "md" / "p-1.md").exists())
            self.assertTrue((out_dir / "json" / "p-1.json").exists())
            self.assertTrue((out_dir / "patients_summary.jsonl").exists())
            self.assertTrue((out_dir / "etl_manifest.json").exists())
            self.assertTrue((out_dir / "etl_report.md").exists())

            md_p1 = (out_dir / "md" / "p-1.md").read_text(encoding="utf-8")
            self.assertIn("ID: p-1", md_p1)
            self.assertIn("Hypertension", md_p1)
            self.assertIn("Lisinopril 10 MG (started 2020-03-01, x3)", md_p1)
            self.assertIn("Systolic Blood Pressure: 145 mm[Hg] (2024-06-15) [range 128.0..145.0, n=3]", md_p1)
            # SSN should not leak into clinical summary
            self.assertNotIn("999-00-1111", md_p1)

            # Verify JSONL has 2 valid JSON lines
            jsonl_lines = (out_dir / "patients_summary.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(jsonl_lines), 2)
            parsed = json.loads(jsonl_lines[0])
            self.assertIn("clinical_summary_md", parsed)
            self.assertIn("estimated_tokens", parsed)

            # Verify max_tokens budget trims without error
            out_trimmed = tmp_path / "out_trimmed"
            manifest_trim = run_etl(
                input_dir=in_dir,
                output_dir=out_trimmed,
                max_tokens=40,
                limit=1,
            )
            self.assertEqual(manifest_trim["patients_processed"], 1)
            self.assertEqual(manifest_trim["truncated_patients"], 1)
            self.assertLessEqual(manifest_trim["tokens"]["max"], 45)

    def test_fhir_etl_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "fhir"
            out_dir = tmp_path / "out"
            in_dir.mkdir()

            bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "fhir-p1",
                            "gender": "female",
                            "birthDate": "1985-08-20",
                        }
                    },
                    {
                        "resource": {
                            "resourceType": "Condition",
                            "id": "c1",
                            "subject": {"reference": "urn:uuid:fhir-p1"},
                            "onsetDateTime": "2023-04-01T10:00:00Z",
                            "code": {"coding": [{"code": "44054006", "display": "Type 2 diabetes"}]},
                        }
                    },
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "id": "o1",
                            "subject": {"reference": "Patient/fhir-p1"},
                            "effectiveDateTime": "2024-05-01T08:00:00Z",
                            "code": {"coding": [{"code": "4548-4", "display": "Hemoglobin A1c"}]},
                            "valueQuantity": {"value": 7.2, "unit": "%"},
                        }
                    },
                ],
            }
            (in_dir / "patient1.json").write_text(json.dumps(bundle), encoding="utf-8")

            manifest = run_etl(input_dir=in_dir, output_dir=out_dir, data_format="auto")
            self.assertEqual(manifest["input_format"], "fhir")
            self.assertEqual(manifest["patients_processed"], 1)
            md_text = (out_dir / "md" / "fhir-p1.md").read_text(encoding="utf-8")
            self.assertIn("Type 2 diabetes", md_text)
            self.assertIn("Hemoglobin A1c: 7.2 % (2024-05-01)", md_text)


if __name__ == "__main__":
    unittest.main()
