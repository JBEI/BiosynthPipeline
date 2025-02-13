#!/usr/bin/python

from demo_repo.counter import Counter


class ServiceException(Exception):
    """Basic exception for the Service class."""

    pass


class Service:
    """Simple service that uses the Counter class."""

    def __init__(self):
        """Set up service object counters."""
        # For better testability, we might pass these in as parameters.
        self._service1_counter = Counter()
        self._service2_counter = Counter()

    def service1(self):
        """First dummy service call for demo purposes."""
        self._service1_counter.increment()

    def service2(self):
        """Second dummy service for demo purposes."""
        self._service2_counter.increment()

    def num_calls(self):
        """Returns number of calls for two services."""
        return self._service1_counter.get_count(), self._service2_counter.get_count()

    def fail(self):
        """Failing service call."""
        raise ServiceException("Failed call.")


def main():
    print("Meant to be used as a library.")


if __name__ == "__main__":
    main()
