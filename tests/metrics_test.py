"""
Unit tests for dagqa/eval/metrics.py

Run with:
    pytest tests/metrics_test.py -v
Run with docker:
    Setup:
        docker compose --profile test build test
    Run Tests:
        docker compose --profile test run --rm test
For the cosine_sim tests that involve the neural model, _sentence_transformer is
mocked so no model download or GPU is required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dagqa.eval.metrics import answer_f1, cosine_sim, exact_match

# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical_strings(self):
        """Identical input → perfect match."""
        assert exact_match("Paris", "Paris", "") == 1.0

    def test_completely_different(self):
        """No similarity at all → no match."""
        assert exact_match("Paris", "London", "") == 0.0

    def test_case_insensitive(self):
        """Normalisation lowercases before comparing."""
        assert exact_match("Paris", "paris", "") == 1.0

    def test_leading_trailing_whitespace(self):
        """Extra whitespace is collapsed by the normaliser."""
        assert exact_match("  Paris  ", "Paris", "") == 1.0

    def test_article_stripped(self):
        """'a', 'an', 'the' are removed before comparing."""
        assert exact_match("The Eiffel Tower", "Eiffel Tower", "") == 1.0

    def test_punctuation_stripped(self):
        """Punctuation is removed before comparing."""
        assert exact_match("hello!", "hello", "") == 1.0

    def test_both_empty_strings(self):
        """Two empty strings are equal."""
        assert exact_match("", "", "") == 1.0

    def test_one_empty_string(self):
        """Empty prediction vs non-empty ground truth → no match."""
        assert exact_match("", "Paris", "") == 0.0


# ---------------------------------------------------------------------------
# answer_f1
# ---------------------------------------------------------------------------


class TestAnswerF1:
    def test_identical_sentences(self):
        """Perfect token overlap → F1 of 1.0."""
        assert answer_f1("cat sat mat", "cat sat mat", "") == 1.0

    def test_no_token_overlap(self):
        """Zero tokens in common → F1 of 0.0."""
        assert answer_f1("cat", "dog", "") == 0.0

    def test_both_empty(self):
        """Both empty → treated as equal (special branch in implementation)."""
        assert answer_f1("", "", "") == 1.0

    def test_prediction_empty(self):
        """Empty prediction, non-empty ground truth → F1 of 0.0."""
        assert answer_f1("", "cat", "") == 0.0

    def test_ground_truth_empty(self):
        """Non-empty prediction, empty ground truth → F1 of 0.0."""
        assert answer_f1("cat", "", "") == 0.0

    def test_partial_overlap(self):
        """Prediction is a strict subset of ground truth.

        pred:  ['cat', 'sat']          (2 tokens)
        gt:    ['cat', 'sat', 'mat']   (3 tokens)
        common: 2
        precision = 2/2 = 1.0
        recall    = 2/3
        F1        = 2 * 1.0 * (2/3) / (1.0 + 2/3) = 0.8
        """
        assert answer_f1("cat sat", "cat sat mat", "") == pytest.approx(0.8)

    def test_same_words_different_order(self):
        """F1 is bag-of-words — word order does not matter."""
        assert answer_f1("A B C", "C B A", "") == 1.0

    def test_repeated_tokens_in_prediction(self):
        """Counter intersection handles duplicates correctly.

        pred:  ['cat', 'cat']  → Counter({'cat': 2})
        gt:    ['cat']         → Counter({'cat': 1})
        common: Counter({'cat': 1}), num_same = 1
        precision = 1/2 = 0.5
        recall    = 1/1 = 1.0
        F1        = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3 ≈ 0.6667
        """
        assert answer_f1("cat cat", "cat", "") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------


class TestCosineSim:
    """
    Cases 17-20 hit fast-path branches before the model is ever called.
    Cases 21-23 mock _sentence_transformer to inject known embeddings,
    which lets us verify the dot-product normalisation math in isolation
    without loading or downloading the real model.
    """

    # --- fast-path cases (no mock needed) -----------------------------------

    def test_identical_strings_fast_path(self):
        """Identical strings short-circuit to 1.0 immediately."""
        assert cosine_sim("hello", "hello", "") == 1.0

    def test_empty_prediction_fast_path(self):
        """Empty prediction short-circuits to 0.0 immediately."""
        assert cosine_sim("", "hello", "") == 0.0

    def test_empty_ground_truth_fast_path(self):
        """Empty ground truth short-circuits to 0.0 immediately."""
        assert cosine_sim("hello", "", "") == 0.0

    def test_both_empty_fast_path(self):
        """Both empty — the '' == '' branch fires first, returning 1.0."""
        assert cosine_sim("", "", "") == 1.0

    # --- math cases (mock the model) ----------------------------------------

    def _make_mock(self, embeddings: np.ndarray) -> MagicMock:
        """Return a fake SentenceTransformer whose encode() returns `embeddings`."""
        fake_model = MagicMock()
        fake_model.encode.return_value = embeddings
        return fake_model

    def test_known_embeddings_partial_similarity(self):
        """Verify the normalisation and dot-product arithmetic.

        vec_a = [1, 0]   -> already unit length
        vec_b = [1, 1]   -> normalised to [1/sqrt(2), 1/sqrt(2)]
        expected cosine  = dot([1,0], [1/sqrt(2), 1/sqrt(2)]) = 1/sqrt(2) ~= 0.7071
        """
        embeddings = np.array([[1.0, 0.0], [1.0, 1.0]])
        fake_model = self._make_mock(embeddings)

        with patch("dagqa.eval.metrics._sentence_transformer", return_value=fake_model):
            result = cosine_sim("anything", "something else", "")

        assert result == pytest.approx(1 / np.sqrt(2), abs=1e-5)

    def test_orthogonal_embeddings_zero_similarity(self):
        """Perpendicular vectors → cosine similarity of 0.0.

        vec_a = [1, 0], vec_b = [0, 1]
        dot product = 0
        """
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        fake_model = self._make_mock(embeddings)

        with patch("dagqa.eval.metrics._sentence_transformer", return_value=fake_model):
            result = cosine_sim("anything", "something else", "")

        assert result == pytest.approx(0.0, abs=1e-5)

    def test_parallel_embeddings_perfect_similarity(self):
        """Vectors pointing in the same direction → cosine similarity of 1.0.

        vec_a = [1, 2], vec_b = [2, 4]  (same direction, different magnitude)
        after normalisation both become [1/sqrt(5), 2/sqrt(5)]
        dot product = 1
        """
        embeddings = np.array([[1.0, 2.0], [2.0, 4.0]])
        fake_model = self._make_mock(embeddings)

        with patch("dagqa.eval.metrics._sentence_transformer", return_value=fake_model):
            result = cosine_sim("anything", "something else", "")

        assert result == pytest.approx(1.0, abs=1e-5)
