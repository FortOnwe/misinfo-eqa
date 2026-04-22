import unittest

from misinfo_eqa.schema import Example
from misinfo_eqa.stressors import (
    analyze_label_rationale_mismatches,
    evidence_relevance,
    flag_label_rationale_mismatches,
    is_ambiguous,
    mask_keywords,
    top_label_keywords,
)


def ex(idx, claim, label, evidence=""):
    return Example(
        id=str(idx),
        dataset="demo",
        split="train",
        claim=claim,
        evidence_text=evidence,
        label3=label,
        date="2024-01-01",
        source_url="",
        metadata={},
    )


class StressorTests(unittest.TestCase):
    def test_keyword_mining_and_masking(self):
        examples = [
            ex(1, "official vaccine data confirmed accurate", "true"),
            ex(2, "official records confirmed accurate", "true"),
            ex(3, "viral hoax was debunked fake", "false"),
            ex(4, "viral hoax was fake", "false"),
        ]
        keywords = top_label_keywords(examples, top_k=3)
        self.assertIn("official", keywords["true"])
        masked = mask_keywords("official records confirmed accurate", {"official"})
        self.assertEqual(masked, "[MASK] records confirmed accurate")

    def test_ambiguity(self):
        self.assertTrue(is_ambiguous(ex(1, "This is reportedly disputed", "true")))
        self.assertTrue(is_ambiguous(ex(2, "Plain claim", "unknown")))
        self.assertFalse(is_ambiguous(ex(3, "Plain claim", "true")))

    def test_mismatch_flags(self):
        flagged = flag_label_rationale_mismatches(
            [
                ex(1, "The city banned cars", "true", "No evidence supports this false claim."),
                ex(2, "The law passed", "true", "The law passed with official confirmation."),
            ]
        )
        self.assertTrue(flagged)
        self.assertIn("heuristic_label_disagrees:false", flagged[0]["reasons"])

    def test_sentence_level_relevance_avoids_long_evidence_false_positive(self):
        relevance = evidence_relevance(
            "photograph shows eleven original staff members of microsoft in 1978.",
            (
                "in 1978, microsoft had just completed its first million-dollar sales year. "
                "all eleven of the employees in the albuquerque picture were about to make "
                "the trip to seattle. the rest of this long article adds background context."
            ),
        )
        self.assertGreaterEqual(relevance["score"], 0.18)

    def test_unrelated_evidence_gets_low_relevance_flag(self):
        flagged = flag_label_rationale_mismatches(
            [
                ex(
                    1,
                    "a glacier is growing again due to climate change.",
                    "unknown",
                    "Question marks are punctuation symbols used in written language.",
                )
            ]
        )
        self.assertTrue(flagged)
        self.assertIn("low_claim_evidence_relevance", flagged[0]["reasons"])
        self.assertLess(flagged[0]["relevance_score"], 0.18)

    def test_long_relevant_evidence_suppresses_standalone_label_cue_disagreement(self):
        evidence = (
            "Official report says the policy passed after a public vote. "
            "The article provides context about the meeting, earlier drafts, public debate, "
            "and unrelated background details. "
            "Additional paragraphs discuss the committee history, the mayor's comments, "
            "implementation timelines, and several reactions from local residents. "
            "More context follows to make this a long narrative evidence passage with enough "
            "claim overlap to be relevant but too much surrounding prose for a simple cue "
            "classifier to be trusted by itself. The final section includes dates, names, "
            "secondary reactions, publication history, and additional article framing that "
            "does not change the central evidence but makes the cue-based label guess noisy."
        )
        flagged = flag_label_rationale_mismatches(
            [ex(1, "Official report says the policy passed.", "false", evidence)]
        )
        self.assertFalse(flagged)

    def test_mismatch_analysis_counts_beyond_storage_limit(self):
        analysis = analyze_label_rationale_mismatches(
            [
                ex(1, "The city banned cars", "true", "No evidence supports this false claim."),
                ex(2, "The school closed", "true", "No evidence supports this false claim."),
                ex(3, "The policy changed", "true", "No evidence supports this false claim."),
            ],
            limit=2,
        )
        self.assertEqual(analysis["n_flagged"], 3)
        self.assertEqual(analysis["examples_stored"], 2)
        self.assertTrue(analysis["examples_truncated"])
        self.assertAlmostEqual(analysis["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
