import json
import unittest

from src.judge_client import build_request
from src.judge_contract import (POINTWISE_ITEMS, contract_fingerprint, pairwise_prompt, pairwise_schema,
                                pointwise_prompt, pointwise_schema, sha256_hex)


def strict_compatible(schema):
    """OpenAI strict mode: every object lists all properties as required and forbids extras."""
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            return False
        if sorted(schema.get("required", [])) != sorted(schema["properties"]):
            return False
        return all(strict_compatible(child) for child in schema["properties"].values())
    if schema.get("type") == "array":
        return strict_compatible(schema["items"])
    return True


QUESTION = {"question": "질문?", "options": ["가", "나", "다", "라"], "correctOptionIndex": 1,
            "explanation": "설명", "sourceEvidence": "근거",
            # Fields that must never reach a Judge prompt.
            "model": "gpt-5.4-mini", "promptVersion": "pilot-v1", "questionReviews": [{"answerAccuracy": "fail"}],
            "validatorStatus": "fail"}


class JudgeContractTest(unittest.TestCase):
    def test_schemas_satisfy_openai_strict_shape(self):
        self.assertTrue(strict_compatible(pointwise_schema()))
        self.assertTrue(strict_compatible(pairwise_schema()))
        properties = pointwise_schema()["properties"]["questions"]["items"]["properties"]
        self.assertEqual(set(properties), {"questionIndex", *POINTWISE_ITEMS})

    def test_prompts_keep_original_index_and_hide_identity_and_human_data(self):
        prompt = pointwise_prompt("contentText 본문", {2: QUESTION})
        self.assertIn('"questionIndex": 2', prompt)
        pair = pairwise_prompt("contentText 본문", {0: QUESTION}, {0: QUESTION})
        for text in (prompt, pair):
            for hidden in ("gpt-5.4-mini", "gemini", "pilot-v1", "questionReviews", "validatorStatus",
                           "answerAccuracy", "Gemini", "OpenAI"):
                self.assertNotIn(hidden, text)
        self.assertIn("[SET_1]", pair)
        self.assertIn("[SET_2]", pair)
        self.assertIn("TIE", pair)

    def test_requests_set_medium_reasoning_and_strict_schema(self):
        openai = build_request("openai", "gpt-6.1-sol", "pointwise", "p", pointwise_schema(), "medium", 100)
        self.assertEqual(openai["reasoning"], {"effort": "medium"})
        self.assertTrue(openai["text"]["format"]["strict"])
        gemini = build_request("gemini", "gemini-3.8-flash", "pairwise", "p", pairwise_schema(), "medium", 100)
        self.assertEqual(gemini["generation_config"], {"max_output_tokens": 100, "thinking_level": "medium"})
        self.assertEqual(gemini["response_format"]["mime_type"], "application/json")
        with self.assertRaises(ValueError):
            build_request("openai", "gpt-6.1-sol", "pointwise", "p", pointwise_schema(), "medium", None)

    def test_identity_hashes_are_deterministic(self):
        self.assertEqual(contract_fingerprint(), contract_fingerprint())
        self.assertEqual(sha256_hex({"b": 1, "a": "가"}), sha256_hex({"a": "가", "b": 1}))
        with self.assertRaises(ValueError):
            sha256_hex({"value": float("nan")})
        self.assertEqual(json.loads(json.dumps(pointwise_schema())), pointwise_schema())


if __name__ == "__main__":
    unittest.main()
