from demo_repo.service import Service


def test_service_call_counts():
    """Tests counts based on number of calls to both services.

    Uses both the counter and service modules integrated together.
    """
    service = Service()
    for _i in range(3):
        service.service1()
    for _i in range(8):
        service.service2()
    for _i in range(2):
        service.service1()
    for _i in range(6):
        service.service2()
    assert service.num_calls() == (3 + 2, 8 + 6)
