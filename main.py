import matplotlib

matplotlib.use('TkAgg')  # Принудительно отключаем перехват графиков PyCharm'ом
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import config
from environment import SimulationEnvironment
from fuzzy_controller import FuzzyPathPlanner


def main():
    env = SimulationEnvironment()

    # Инициализируем контроллер с параметрами, найденными через Optuna
    planner = FuzzyPathPlanner(
        close_max=1.88,
        far_min=3.76,
        slow_peak=0.22,
        med_peak=0.52,
        fast_peak=0.86
    )
    # Настройка графики
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Построение пути с нечёткой логикой", fontsize=14)

    robot_circle = plt.Circle(env.robot_pos, config.ROBOT_RADIUS, color='blue', label='Робот')
    target_circle = plt.Circle(env.target_pos, config.TARGET_RADIUS, color='lime', label='Цель')
    ax.add_patch(robot_circle)
    ax.add_patch(target_circle)

    obs_patches = [plt.Circle(o['pos'], config.OBSTACLE_RADIUS, color='purple', alpha=0.7) for o in env.obstacles]
    for p in obs_patches:
        ax.add_patch(p)

    trail_line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.6, label='Траектория')
    trail_x, trail_y = [], []

    # Инициализация стрелки направления (dummy data)
    direction_arrow = ax.arrow(0, 0, 0, 0, head_width=0.4, color='cyan', alpha=0.9)
    ax.legend(loc='upper left', fontsize=10)

    step_counter = [0]  # Используем список для изменения внутри функции

    def update(frame):
        step_counter[0] += 1

        # Получаем данные от сенсоров
        d_left, d_front, d_right = env.get_sensor_distances()
        t_dir = env.get_target_angle()

        # Нечеткий вывод
        turn_angle, speed_val = planner.compute_action(d_left, d_front, d_right, t_dir, step_id=step_counter[0])

        # Симуляция физики
        env.step(turn_angle, speed_val)

        # Обновление графики
        for i, p in enumerate(obs_patches):
            p.center = env.obstacles[i]['pos']

        robot_circle.center = env.robot_pos
        trail_x.append(env.robot_pos[0])
        trail_y.append(env.robot_pos[1])
        trail_line.set_data(trail_x, trail_y)

        nonlocal direction_arrow
        direction_arrow.remove()
        direction_arrow = ax.arrow(
            env.robot_pos[0], env.robot_pos[1],
            1.8 * np.cos(env.robot_angle), 1.8 * np.sin(env.robot_angle),
            head_width=0.4, color='cyan', alpha=0.9, length_includes_head=True
        )

        # Проверка завершения
        if env.is_goal_reached():
            ax.set_title('ЦЕЛЬ ДОСТИГНУТА!', color='green', fontsize=16)
            ani.event_source.stop()
        elif step_counter[0] > 2000:
            ax.set_title('ПРЕВЫШЕНО ВРЕМЯ', color='red', fontsize=16)
            ani.event_source.stop()
        else:
            ax.set_title(
                f'Шаг: {step_counter[0]} | Поворот: {turn_angle:+.1f}° | Скорость: {np.linalg.norm(env.robot_vel):.2f}')

        return [robot_circle, trail_line, direction_arrow] + obs_patches

    ani = FuncAnimation(fig, update, frames=2000, interval=50, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()