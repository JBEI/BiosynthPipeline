import mock
import pytest

from demo_repo.service import Service, ServiceException


def test_new_service():
    """Tests state of new service."""
    service = Service()
    assert service.num_calls() == (0, 0)


@mock.patch("demo_repo.counter.Counter.increment", return_value=None)
def test_service1(increment_method):
    """Tests calls to service 1 increments counts."""
    service = Service()
    service.service1()
    increment_method.assert_called_once_with()


@mock.patch("demo_repo.counter.Counter.increment", return_value=None)
def test_service2(increment_method):
    """Tests calls to service 2 increments counts."""
    service = Service()
    service.service2()
    increment_method.assert_called_once_with()


@mock.patch("demo_repo.counter.Counter.get_count", return_value=1)
def test_num_calls(get_count):
    """Tests calls to get number of service calls."""
    service = Service()
    assert service.num_calls() == (1, 1)
    assert get_count.call_count == 2


def test_fail():
    """Tests that we get an expected exception."""
    service = Service()
    try:
        service.fail()
        pytest.fail("Expected an exception.")
    except ServiceException as e:
        assert "Failed call." in str(e)
