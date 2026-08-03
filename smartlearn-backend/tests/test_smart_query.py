import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.smart_query import (
    SmartQueryPlan,
    build_document_context,
    build_planner_prompt,
    merge_ranked_hits,
    parse_query_plan,
    plan_smart_query,
)


class SmartQueryTests(unittest.TestCase):
    def test_planner_preserves_ai_generated_semantic_query(self):
        plan = parse_query_plan(
            """```json
            {
              "intent": "document_metadata.author",
              "canonical_query": "paper authors and affiliations",
              "search_queries": ["title page author names"],
              "answer_language": "zh",
              "confidence": 0.97
            }
            ```""",
            original_question="这篇文章是谁写的？",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, "document_metadata.author")
        self.assertEqual(plan.canonical_query, "paper authors and affiliations")
        self.assertIn("这篇文章是谁写的？", plan.search_queries)

    def test_prompt_uses_bounded_document_clues_for_reference_resolution(self):
        pages = [
            {"page": 1, "text": "A Paper Title Alice Example Bob Example"},
            {"page": 2, "text": "Introduction to the retrieval method"},
        ]
        context = build_document_context(pages)
        prompt = build_planner_prompt(
            question="这个方法有什么限制？",
            history=[],
            retrieval_language="English",
            document_context=context,
        )

        self.assertIn("A Paper Title", prompt)
        self.assertIn("Replace pronouns", prompt)
        self.assertIn("这个方法有什么限制？", prompt)

    def test_invalid_json_falls_back_to_none(self):
        self.assertIsNone(parse_query_plan("not json", "question"))

    def test_rank_fusion_rewards_hits_found_by_multiple_queries(self):
        first = [
            {"chunk_id": "a", "page": 1, "score": 0.7},
            {"chunk_id": "b", "page": 2, "score": 0.6},
        ]
        second = [
            {"chunk_id": "b", "page": 2, "score": 0.65},
            {"chunk_id": "c", "page": 3, "score": 0.68},
        ]

        merged = merge_ranked_hits([first, second], top_k=3)

        self.assertEqual(merged[0]["chunk_id"], "b")
        self.assertEqual(merged[0]["score"], 0.65)

    def test_rank_fusion_returns_only_one_hit_per_page(self):
        first = [
            {"chunk_id": "a", "page": 1, "score": 0.8},
            {"chunk_id": "b", "page": 1, "score": 0.7},
        ]
        second = [{"chunk_id": "c", "page": 2, "score": 0.6}]

        merged = merge_ranked_hits([first, second], top_k=3)

        self.assertEqual([hit["page"] for hit in merged], [1, 2])

    def test_planner_uses_valid_deepseek_response(self):
        captured_request = {}
        content = """{
          "intent": "document_metadata.author",
          "canonical_query": "Who wrote this?",
          "search_queries": ["authors"],
          "answer_language": "zh",
          "confidence": 0.95
        }"""

        def create_completion(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=content))
                ]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=create_completion
                )
            )
        )

        with patch("openai.OpenAI", return_value=fake_client):
            plan = plan_smart_query(
                "这篇文章是谁写的？",
                pages=[{"page": 1, "text": "English text " * 20}],
                api_key="test-key",
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, "document_metadata.author")
        self.assertEqual(plan.canonical_query, "Who wrote this?")
        self.assertGreaterEqual(captured_request["max_tokens"], 800)

    def test_smart_mode_runs_all_planned_queries(self):
        from services import rag

        plan = SmartQueryPlan(
            intent="document_metadata.author",
            canonical_query="document authors and affiliations",
            search_queries=("document authors and affiliations", "authors"),
            answer_language="zh",
            confidence=0.95,
        )
        hit = {
            "chunk_id": "chunk-0001",
            "page": 2,
            "text": "AUTHORS Alice Example",
            "matched_text": "AUTHORS Alice Example",
            "score": 0.8,
            "raw_score": 0.8,
            "lexical_score": 0.5,
        }
        document = {"pages": [], "history": []}

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("services.smart_query.plan_smart_query", return_value=plan),
            patch("services.rag.search_document", side_effect=[[hit], [hit]]) as search,
            patch(
                "services.rag._llm_answer_from_hits",
                return_value="作者是 Alice Example [Page 2]。",
            ),
        ):
            result = rag.answer_document(
                document,
                "这篇文章是谁写的？",
                smart_mode=True,
            )

        self.assertEqual(search.call_count, 2)
        self.assertEqual(result["citations"], [2])
        self.assertEqual(result["sources"][0]["page"], 2)


if __name__ == "__main__":
    unittest.main()
