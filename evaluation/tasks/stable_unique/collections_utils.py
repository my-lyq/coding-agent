def stable_unique(items: list[int]) -> list[int]:
    return sorted(set(items))  # BUG: sorts instead of preserving order
