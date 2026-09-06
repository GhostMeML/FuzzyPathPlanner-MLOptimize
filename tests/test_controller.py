import pytest
import numpy as np
from fuzzy_controller import FuzzyPathPlanner


def test_controller_initialization():
    planner = FuzzyPathPlanner()
    assert planner is not None


def test_nan_input_handling():
    """Тест защиты от некорректных данных (NaN), возвращаемых сбойным сенсором"""
    planner = FuzzyPathPlanner()
    # Передаем NaN вместо дистанции
    turn, speed = planner.compute_action(np.nan, 2.0, 5.0, 10.0)

    # Контроллер должен отработать Fallback и вернуть безопасные значения (0.0), а не упасть
    assert not np.isnan(turn)
    assert not np.isnan(speed)


def test_out_of_bounds_input():
    """Тест валидации экстремальных значений (Infinity)"""
    planner = FuzzyPathPlanner()
    turn, speed = planner.compute_action(float('inf'), -50.0, 3.0, 999.0)

    # Углы и скорости должны оставаться в пределах логики физики
    assert -90 <= turn <= 90
    assert 0.0 <= speed <= 1.0