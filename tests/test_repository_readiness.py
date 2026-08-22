from pathlib import Path

import numpy as np
import yaml
from streamlit.testing.v1 import AppTest

from auxiliar_functions import checker
from optimizer import basic_use_brute_force, basic_use_iterative, basic_use_tn

ROOT = Path(__file__).resolve().parents[1]


def test_documented_optimizers_match_the_brute_force_solution() -> None:
    conditions = [
        [None, 0, 0, None, 1],
        [1, 2, 0, 2, None],
        [None, None, 1, 1, 2],
    ]
    targets = [[0, 1], [4, 1], [1, 0]]
    constraints = [conditions, targets]
    costs = [
        np.array(machine_costs)
        for machine_costs in [
            [1.1, 2.71],
            [2.16, 5.3, 4.21],
            [2.2, 1.75],
            [4.5, 7.1, 1.05],
            [9.1, 5.1, 0.77],
        ]
    ]
    expected_solution = np.array([0, 0, 1, 2, 2])
    expected_cost = 6.83

    results = [
        basic_use_brute_force(constraints, costs),
        basic_use_tn(constraints, costs, 0.25),
        basic_use_iterative(constraints, costs, 0.25),
    ]

    for solution, cost in results:
        assert np.array_equal(solution, expected_solution)
        assert np.isclose(cost, expected_cost)
        assert checker(constraints, solution) == (True, -1)


def test_streamlit_application_starts_without_exceptions() -> None:
    application = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()

    assert not list(application.exception)


def test_frozen_repository_pauses_routine_dependency_updates() -> None:
    assert not (ROOT / ".github" / "dependabot.yml").exists()

    maintenance = (ROOT / "MAINTENANCE.md").read_text(encoding="utf-8")
    assert "frozen for routine maintenance" in maintenance
    assert "reviewer or editor requests" in maintenance


def test_required_gate_depends_on_the_real_quality_job() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]

    assert set(jobs) == {"quality", "required-pr-ci"}
    assert jobs["required-pr-ci"]["needs"] == ["quality"]
    assert jobs["required-pr-ci"]["if"] == "${{ always() }}"
