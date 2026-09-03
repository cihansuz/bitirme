import pytest
from src.planner.pulp_optimizer import NutritionalPlanner


def test_nutritional_planner_optimization():
    planner = NutritionalPlanner()

    constraints = {
        "carbohydrate_ratio_max": 0.45,
        "minimum_fiber_g": 30.0,
        "protein_ratio_max": 0.25,
        "min_iron_mg": 18.0,
        "min_vitamin_c_mg": 75.0
    }

    plan = planner.plan_macros(constraints=constraints, target_calories=1800.0)

    assert plan["solver_status"] in ("Optimal", "1")
    assert plan["total_calories"] >= 1600.0 and plan["total_calories"] <= 2000.0
    assert plan["fiber_g"] >= 28.0
    assert plan["carb_ratio"] <= 46.0
    assert len(plan["suggested_portions"]) > 0
