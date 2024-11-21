#!/usr/bin/python


class Counter:
    """Simple counter for demonstrating tests."""

    def __init__(self):
        """Set up counter."""
        self._count = 0

    def increment(self):
        """Increments the  count."""
        self._count += 1

    def get_count(self):
        """Returns the count."""
        return self._count


def main():
    print("Meant to be used as a library.")


if __name__ == "__main__":
    main()
