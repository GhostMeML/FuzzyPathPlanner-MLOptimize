import optuna
import numpy as np
import skfuzzy as fuzz
from environment import SimulationEnvironment
from fuzzy_controller import FuzzyPathPlanner


def objective(trial):
    # 1. Настройка сенсоров и скорости
    close_max = trial.suggest_float('close_max', 1.5, 3.5)
    far_min = trial.suggest_float('far_min', 3.6, 5.5)

    slow_peak = trial.suggest_float('slow_peak', 0.1, 0.4)
    med_peak = trial.suggest_float('med_peak', 0.4, 0.7)
    fast_peak = trial.suggest_float('fast_peak', 0.7, 1.0)

    # Защита от перекрытий: пики должны идти строго по возрастанию
    if not (slow_peak < med_peak < fast_peak):
        raise optuna.exceptions.TrialPruned()

    env = SimulationEnvironment()

    # 2. Инициализируем контроллер с новыми гиперпараметрами СРАЗУ
    planner = FuzzyPathPlanner(
        close_max=close_max,
        far_min=far_min,
        slow_peak=slow_peak,
        med_peak=med_peak,
        fast_peak=fast_peak
    )

    # 3. Запуск симуляции ("в фоне")
    max_steps = 1000
    steps = 0
    collisions = 0

    while steps < max_steps and not env.is_goal_reached():
        d_left, d_front, d_right = env.get_sensor_distances()
        t_dir = env.get_target_angle()

        turn, speed = planner.compute_action(d_left, d_front, d_right, t_dir)

        prev_vel = np.linalg.norm(env.robot_vel)
        env.step(turn, speed)

        if np.linalg.norm(env.robot_vel) < prev_vel * 0.5:
            collisions += 1

        steps += 1

    # 4. Фитнес-функция
    distance_to_goal = np.linalg.norm(env.robot_pos - env.target_pos)
    score = (distance_to_goal * 20.0) + (collisions * 100.0) + (steps * 0.1)

    if not env.is_goal_reached():
        score += 500.0

    return score

if __name__ == "__main__":
    print("Запуск ML-оптимизатора (Сенсоры + Скорость)...")
    study = optuna.create_study(direction='minimize')

    # timeout=120 ограничит поиск двумя минутами, чтобы не ждать вечно
    study.optimize(objective, n_trials=20, timeout=120)

    print("\n=== Идеальный баланс найден ===")
    print("Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.2f}")