"""Pytest configuration shared by the test suite."""


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: end-to-end check on a small molecule subset (runs in well under a minute)")
