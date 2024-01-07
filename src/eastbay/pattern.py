class Pattern:
    """Class for parsing PSMC-style pattern strings."""

    def __init__(self, pattern: str):
        try:
            epochs = self._epochs = []
            for s in pattern.split("+"):
                if "*" in s:
                    k, width = map(int, s.split("*"))
                else:
                    k = 1
                    width = int(s)
                epochs += [width] * k
        except Exception:
            raise ValueError("could not parse pattern")
        if len(epochs) == 0:
            raise ValueError("pattern must contain at least one epoch")
        if any(e <= 0 for e in epochs):
            raise ValueError("epochs must be positive")

    @property
    def M(self):
        return sum(self._epochs)

    def __len__(self) -> int:
        return len(self._epochs)

    def expand(self, x):
        assert len(x) == len(self)
        return sum([e * [xx] for e, xx in zip(self._epochs, x)], [])
