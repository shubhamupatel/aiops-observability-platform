"""Smoke test that the installed package can be imported."""


def test_simulator_package_imports():
    import aiops.simulator	# fails here if the package isn't installed into the venv