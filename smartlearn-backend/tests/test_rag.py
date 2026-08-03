import unittest

from services import rag


class RagPureHelperTests(unittest.TestCase):
    def test_fingerprint_hashes_text_after_first_200_characters(self):
        pages_a = [{"page": 1, "text": "A" * 250 + "X" + "B" * 249}]
        pages_b = [{"page": 1, "text": "A" * 250 + "Y" + "B" * 249}]

        self.assertNotEqual(
            rag._content_fingerprint(pages_a),
            rag._content_fingerprint(pages_b),
        )

    def test_generic_paper_question_is_not_metadata_question(self):
        self.assertFalse(
            rag.is_document_meta_question(
                "What is the name of the sparse retriever used in this paper?"
            )
        )
        self.assertFalse(
            rag.is_document_meta_question("这篇论文使用的稀疏检索器是什么？")
        )
        self.assertTrue(rag.is_document_meta_question("What is the paper title?"))

    def test_document_reference_is_not_mistaken_for_followup(self):
        self.assertFalse(rag.is_followup_question("What is the title of this paper?"))
        self.assertTrue(rag.is_followup_question("Give one more detail from that page."))

    def test_citations_must_come_from_current_hits(self):
        answer = "Supported [Page 6], invented [Page 999]."
        self.assertEqual(rag.extract_citations(answer, [{"page": 6}]), [6])
        self.assertEqual(rag.extract_citations("No citation", [{"page": 6}]), [])

    def test_adjacent_chunk_overlap_is_removed(self):
        left = "prefix " + "shared text " * 10
        overlap = left[-80:]
        right = overlap + "suffix"

        merged = rag._merge_overlapping_text(left, right)

        self.assertEqual(merged.count(overlap), 1)
        self.assertTrue(merged.endswith("suffix"))


if __name__ == "__main__":
    unittest.main()
