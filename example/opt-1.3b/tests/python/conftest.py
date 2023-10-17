import pytest

def pytest_addoption(parser):
    parser.addoption("--num_workers", action="store", default=2)
    parser.addoption("--num_dlls", action="store", default=2)
    

@pytest.fixture
def num_workers(request):
    return request.config.getoption("--num_workers")

@pytest.fixture
def num_dlls(request):
    return request.config.getoption("--num_dlls")