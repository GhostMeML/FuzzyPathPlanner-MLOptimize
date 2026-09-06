import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import config
import logging
import json
import math

# Настройка JSON-логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')


class FuzzyPathPlanner:
    # Добавляем гиперпараметры со значениями по умолчанию
    def __init__(self, close_max=2.0, far_min=4.0, slow_peak=0.4, med_peak=0.7, fast_peak=1.0):
        self.close_max = close_max
        self.far_min = far_min
        self.slow_peak = slow_peak
        self.med_peak = med_peak
        self.fast_peak = fast_peak

        self.dist_left = ctrl.Antecedent(np.arange(0, config.SENSOR_RANGE + 0.1, 0.1), 'dist_left')
        self.dist_front = ctrl.Antecedent(np.arange(0, config.SENSOR_RANGE + 0.1, 0.1), 'dist_front')
        self.dist_right = ctrl.Antecedent(np.arange(0, config.SENSOR_RANGE + 0.1, 0.1), 'dist_right')
        self.target_dir = ctrl.Antecedent(np.arange(-180, 181, 1), 'target_dir')

        self.turn = ctrl.Consequent(np.arange(-90, 91, 1), 'turn')
        self.speed = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'speed')

        self._setup_membership_functions()
        self._setup_rules()

        self.control_system = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.control_system)

    def _setup_membership_functions(self):
        # Используем переданные гиперпараметры для сенсоров
        for var in (self.dist_left, self.dist_front, self.dist_right):
            var['Близко'] = fuzz.trimf(var.universe, [0, 0, self.close_max])
            var['Средне'] = fuzz.trimf(var.universe, [1, 3, 5])
            var['Далеко'] = fuzz.trimf(var.universe, [self.far_min, 6, 6])

        self.target_dir['Слева'] = fuzz.trimf(self.target_dir.universe, [-180, -180, -45])
        self.target_dir['Прямо'] = fuzz.trimf(self.target_dir.universe, [-45, 0, 45])
        self.target_dir['Справа'] = fuzz.trimf(self.target_dir.universe, [45, 180, 180])

        self.turn['Резко влево'] = fuzz.trimf(self.turn.universe, [-90, -90, -45])
        self.turn['Плавно влево'] = fuzz.trimf(self.turn.universe, [-60, -30, 0])
        self.turn['Прямо'] = fuzz.trimf(self.turn.universe, [-15, 0, 15])
        self.turn['Плавно вправо'] = fuzz.trimf(self.turn.universe, [0, 30, 60])
        self.turn['Резко вправо'] = fuzz.trimf(self.turn.universe, [45, 90, 90])

        # Используем гиперпараметры для настройки скорости
        self.speed['Медленно'] = fuzz.trimf(self.speed.universe, [0.0, 0.0, self.slow_peak])
        self.speed['Средне'] = fuzz.trimf(self.speed.universe,
                                          [self.slow_peak - 0.1, self.med_peak, self.fast_peak + 0.1])
        self.speed['Быстро'] = fuzz.trimf(self.speed.universe, [self.med_peak, self.fast_peak, 1.0])

    def _setup_rules(self):
        # Разбиваем каждое правило с двумя выводами на два независимых правила.
        # Это предотвращает баг с кортежами в networkx и делает граф вычислений абсолютно стабильным.
        self.rules = [
            # Правило 1
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Слева'], self.turn['Резко влево']),
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Слева'], self.speed['Медленно']),

            # Правило 2
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Прямо'], self.turn['Резко влево']),
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Прямо'], self.speed['Медленно']),

            # Правило 3
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Справа'], self.turn['Резко вправо']),
            ctrl.Rule(self.dist_front['Близко'] & self.target_dir['Справа'], self.speed['Медленно']),

            # Правило 4
            ctrl.Rule(self.dist_front['Средне'] & self.dist_left['Близко'] & self.target_dir['Справа'],
                      self.turn['Резко вправо']),
            ctrl.Rule(self.dist_front['Средне'] & self.dist_left['Близко'] & self.target_dir['Справа'],
                      self.speed['Медленно']),

            # Правило 5
            ctrl.Rule(self.dist_front['Средне'] & self.dist_right['Близко'] & self.target_dir['Слева'],
                      self.turn['Резко влево']),
            ctrl.Rule(self.dist_front['Средне'] & self.dist_right['Близко'] & self.target_dir['Слева'],
                      self.speed['Медленно']),

            # Правило 6
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Слева'], self.turn['Плавно влево']),
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Слева'], self.speed['Средне']),

            # Правило 7
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Прямо'], self.turn['Прямо']),
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Прямо'], self.speed['Средне']),

            # Правило 8
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Справа'], self.turn['Плавно вправо']),
            ctrl.Rule(self.dist_front['Средне'] & self.target_dir['Справа'], self.speed['Средне']),

            # Правило 9
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Слева'], self.turn['Плавно влево']),
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Слева'], self.speed['Быстро']),

            # Правило 10
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Прямо'], self.turn['Прямо']),
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Прямо'], self.speed['Быстро']),

            # Правило 11
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Справа'], self.turn['Плавно вправо']),
            ctrl.Rule(self.dist_front['Далеко'] & self.target_dir['Справа'], self.speed['Быстро']),

            # Правило 12
            ctrl.Rule(self.dist_left['Близко'] & ~self.dist_front['Близко'], self.turn['Плавно вправо']),
            ctrl.Rule(self.dist_left['Близко'] & ~self.dist_front['Близко'], self.speed['Средне']),

            # Правило 13
            ctrl.Rule(self.dist_right['Близко'] & ~self.dist_front['Близко'], self.turn['Плавно влево']),
            ctrl.Rule(self.dist_right['Близко'] & ~self.dist_front['Близко'], self.speed['Средне']),

            # Правило 14
            ctrl.Rule(self.dist_left['Близко'] & self.dist_right['Близко'], self.turn['Резко влево']),
            ctrl.Rule(self.dist_left['Близко'] & self.dist_right['Близко'], self.speed['Медленно']),

            # Правило 15
            ctrl.Rule(self.dist_front['Близко'] & self.dist_left['Далеко'] & self.dist_right['Далеко'],
                      self.turn['Резко влево']),
            ctrl.Rule(self.dist_front['Близко'] & self.dist_left['Далеко'] & self.dist_right['Далеко'],
                      self.speed['Медленно'])
        ]
    def _validate_input(self, val, min_val, max_val, default_val):
        """Строгая валидация входных данных (защита от отказов)"""
        if val is None or math.isnan(val) or math.isinf(val):
            return default_val
        return max(min_val, min(max_val, val))

    def compute_action(self, d_left, d_front, d_right, t_dir, step_id=0):
        # Валидация в стиле строгих проверок транзакций
        safe_left = self._validate_input(d_left, 0, config.SENSOR_RANGE, 0.0)
        safe_front = self._validate_input(d_front, 0, config.SENSOR_RANGE, 0.0)
        safe_right = self._validate_input(d_right, 0, config.SENSOR_RANGE, 0.0)
        safe_dir = self._validate_input(t_dir, -180, 180, 0.0)

        self.sim.input['dist_left'] = safe_left
        self.sim.input['dist_front'] = safe_front
        self.sim.input['dist_right'] = safe_right
        self.sim.input['target_dir'] = safe_dir

        try:
            self.sim.compute()
            turn = self.sim.output['turn']
            speed = self.sim.output['speed']

            # Структурированное логирование метрик
            if step_id % 50 == 0:  # Логируем каждый 50-й шаг, чтобы не спамить
                log_data = {
                    "event": "fuzzy_compute",
                    "step": step_id,
                    "inputs": {"front": safe_front, "target_dir": safe_dir},
                    "outputs": {"turn": round(turn, 2), "speed": round(speed, 2)}
                }
                logging.info(json.dumps(log_data))

            return turn, speed
        except Exception as e:
            # Fallback-механизм: если вывод упал, останавливаемся и логируем ошибку (Rollback)
            error_log = {"event": "fuzzy_error", "error": str(e), "fallback_engaged": True}
            logging.error(json.dumps(error_log))
            return 0.0, 0.0