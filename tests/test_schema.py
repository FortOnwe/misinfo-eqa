import unittest

from misinfo_eqa.schema import coerce_example, evidence_text_quality, normalize_label, parse_date


class SchemaTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(normalize_label("mostly true"), "true")
        self.assertEqual(normalize_label("pants-fire"), "false")
        self.assertEqual(normalize_label("unverified"), "unknown")

    def test_parse_date(self):
        self.assertEqual(parse_date("2024-05-17"), "2024-05-17")
        self.assertEqual(parse_date("", {"year": "2021", "month": "03", "day": "09"}), "2021-03-09")
        self.assertEqual(parse_date("sometime in 2020"), "2020-01-01")

    def test_coerce_example(self):
        example = coerce_example(
            {
                "claim": "The claim is verified.",
                "positive_evidence_text": "Official evidence shows the claim is accurate.",
                "veracity": "true",
                "date": "2022",
                "source_url": "https://example.test",
                "split": "train",
            },
            "demo",
            0,
        )
        self.assertEqual(example.dataset, "demo")
        self.assertEqual(example.label3, "true")
        self.assertEqual(example.date, "2022-01-01")
        self.assertIn("Official evidence", example.evidence_text)
        self.assertEqual(example.metadata["evidence_quality"], "natural_language")

    def test_structured_reference_evidence_is_preserved_but_not_used(self):
        example = coerce_example(
            {
                "claim": "Olivia Munn is dead.",
                "evidence": "c(30098, 301119, NA, NA)",
                "veracity": "unknown",
            },
            "fever",
            0,
        )
        self.assertEqual(example.evidence_text, "")
        self.assertEqual(example.metadata["raw_evidence_text"], "c(30098, 301119, NA, NA)")
        self.assertEqual(example.metadata["evidence_quality"], "structured_reference")

    def test_evidence_text_quality(self):
        self.assertEqual(evidence_text_quality("c(30098, NA, NA)"), "structured_reference")
        self.assertEqual(
            evidence_text_quality("Official records verify that the claim is accurate."),
            "natural_language",
        )


if __name__ == "__main__":
    unittest.main()
