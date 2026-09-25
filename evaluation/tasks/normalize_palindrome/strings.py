def is_palindrome(text: str) -> bool:
    return text == text[::-1]  # BUG: no normalization
