"""xorshift32 — the same stream in Python and in the browser demo.

numpy's Generator cannot be reproduced in JavaScript, and a demo that computes
rather than replays has to run the identical stream.
"""
from __future__ import annotations

import math

__all__ = ["XorShift32", "mix"]

_MASK = 0xFFFFFFFF


def mix(seed: int, index: int) -> int:
    x = ((int(seed) & 0xFFFF) << 16) ^ (int(index) & 0xFFFF) ^ 0x5BF03635
    x = (x * 2654435761) & _MASK
    return x ^ (x >> 15)


class XorShift32:
    def __init__(self, seed: int):
        self._s = (int(seed) & _MASK) or 0x9E3779B9

    def next_u32(self) -> int:
        x = self._s
        x ^= (x << 13) & _MASK
        x ^= x >> 17
        x ^= (x << 5) & _MASK
        self._s = x & _MASK
        return self._s

    def random(self) -> float:
        return self.next_u32() / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def randint(self, lo: int, hi: int) -> int:
        if hi <= lo:
            raise ValueError("hi must exceed lo")
        return lo + int(self.random() * (hi - lo))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
