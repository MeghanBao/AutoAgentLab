"""Tests for the upgraded _check_answer logic — no LLM calls required."""

from autoagentlab.evaluator import Evaluator

check = Evaluator._check_answer


# ── Substring match (existing behaviour) ───────────────────────────

def test_exact_substring():
    assert check("The capital is Paris.", "Paris")

def test_case_insensitive():
    assert check("the answer is paris", "Paris")

def test_symbol_match():
    assert check("The formula for water is H2O.", "H2O")

def test_no_match():
    assert not check("Berlin is the capital of Germany.", "Paris")


# ── Pipe-separated alternatives ─────────────────────────────────────

def test_first_alternative_matches():
    assert check("They weigh the same.", "same|equal|neither")

def test_second_alternative_matches():
    assert check("They are equal in weight.", "same|equal|neither")

def test_third_alternative_matches():
    assert check("Neither is heavier.", "same|equal|neither")

def test_no_alternative_matches():
    assert not check("One is heavier.", "same|equal|neither")

def test_pipe_in_expected_with_numbers():
    assert check("The answer is 195.0", "195|195.0")


# ── Numeric tolerance matching ──────────────────────────────────────

def test_integer_in_sentence():
    assert check("The result is 195.", "195")

def test_float_matches_integer():
    assert check("The answer is 195.0", "195")

def test_integer_matches_float_expected():
    assert check("The answer is 45", "45.0")

def test_within_1pct_relative():
    # 78.54 vs 78.53 → difference < 1%
    assert check("The area is approximately 78.54 square units.", "78.53")

def test_number_with_comma_thousands():
    # "9,716" in response should match expected "9716"
    assert check("The product is 9,716.", "9716")

def test_negative_number():
    assert check("The temperature is -10 degrees.", "-10")

def test_decimal_match():
    assert check("0.375 is the decimal form.", "0.375")

def test_numeric_mismatch():
    assert not check("The answer is 200.", "195")

def test_zero_expected():
    assert check("The result is 0.", "0")

def test_scientific_notation_response():
    assert check("Speed of light is 3e8 m/s.", "300000000")


# ── Edge cases ──────────────────────────────────────────────────────

def test_empty_response():
    assert not check("", "Paris")

def test_expected_in_longer_word():
    # "8" should not match "80" via substring but may via numeric
    # numeric: 80 vs 8 → relative diff = 9/8 = 900% > 1% → no match
    assert not check("There are 80 planets.", "8")

def test_multiword_expected():
    assert check("Plants absorb carbon dioxide from the air.", "carbon dioxide")

def test_instruction_following_yes_no():
    assert check("Yes", "Yes")
    assert not check("No", "Yes")
