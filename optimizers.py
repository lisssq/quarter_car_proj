import time
import numpy as np
from scipy.optimize import minimize
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

class Optimizer:
    def __init__(self, model):
        self.model = model
        self.results = []
    
    def run_nelder_mead(self):
        print("\n")
        print("МЕТОД Nelder-Mead")
        print("="*60)
        
        bounds = [(15000, 50000), (1000, 8000)]
        x0 = [30000, 4000]
        
        start_time = time.time()
        
        result = minimize(
            self.model.objective_function,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': 100,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'disp': False
            }
        )
        
        elapsed_time = time.time() - start_time
        
        k2_opt, c2_opt = result.x
        J_opt = result.fun
        
        result_dict = {
            'method': 'Nelder-Mead',
            'k2': k2_opt,
            'c2': c2_opt,
            'J': J_opt,
            'iterations': result.nit,
            'function_calls': result.nfev,
            'time': elapsed_time,
            'success': result.success
        }
        
        self._print_result(result_dict)
        return result_dict
    
    def run_hyperopt(self, max_evals=50):
        print("\n")
        print(f"HYPEROPT")
        
        space = {
            'k2': hp.uniform('k2', 15000, 50000),
            'c2': hp.uniform('c2', 1000, 8000)
        }
        
        def objective_hyperopt(params):
            k2 = params['k2']
            c2 = params['c2']
            loss = self.model.objective_function([k2, c2])
            return {'loss': loss, 'status': STATUS_OK}
        
        start_time = time.time()
        
        trials = Trials()
        best = fmin(
            fn=objective_hyperopt,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
        
        elapsed_time = time.time() - start_time
        
        k2_opt = best['k2']
        c2_opt = best['c2']
        J_opt = trials.best_trial['result']['loss']
        
        result_dict = {
            'method': 'Hyperopt',
            'k2': k2_opt,
            'c2': c2_opt,
            'J': J_opt,
            'iterations': max_evals,
            'function_calls': max_evals,
            'time': elapsed_time,
            'success': True,
            'trials': trials
        }
        
        self._print_result(result_dict)
        return result_dict
    
    def run_optuna(self, n_trials=50):
        print("\n")
        print(f"OPTUNA (испытаний: {n_trials})")
        
        def objective_optuna(trial):
            k2 = trial.suggest_float('k2', 15000, 50000)
            c2 = trial.suggest_float('c2', 1000, 8000)
            return self.model.objective_function([k2, c2])
        
        start_time = time.time()
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective_optuna, n_trials=n_trials)
        
        elapsed_time = time.time() - start_time
        
        k2_opt = study.best_params['k2']
        c2_opt = study.best_params['c2']
        J_opt = study.best_value
        
        result_dict = {
            'method': 'Optuna',
            'k2': k2_opt,
            'c2': c2_opt,
            'J': J_opt,
            'iterations': n_trials,
            'function_calls': n_trials,
            'time': elapsed_time,
            'success': True,
            'study': study
        }
        
        self._print_result(result_dict)
        return result_dict
    
    def _print_result(self, result):
        print(f"Оптимальные параметры:")
        print(f"  k2 = {result['k2']:.0f} Н/м")
        print(f"  c2 = {result['c2']:.0f} Н·с/м")
        print(f"  Целевая функция J = {result['J']:.4f}")
        print(f"  Итераций: {result['iterations']}")
        print(f"  Вызовов функции: {result['function_calls']}")
        print(f"  Время: {result['time']:.2f} с")
    
    def compare_all(self):
        self.results = []

        self.results.append(self.run_nelder_mead())
        self.results.append(self.run_hyperopt(max_evals=50))
        self.results.append(self.run_optuna(n_trials=50))
        
        self._print_summary_table()
        
        return self.results
    
    def _print_summary_table(self):
        print("\n" + "="*60)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        
        print(f"\n{'Метод':<25} {'k2 (Н/м)':<12} {'c2 (Н·с/м)':<12} {'J':<12} {'Итерации':<10} {'Время (с)':<10}")
        print("-" * 85)
        
        for res in self.results:
            print(f"{res['method']:<25} {res['k2']:<12.0f} {res['c2']:<12.0f} "
                  f"{res['J']:<12.4f} {res['iterations']:<10} {res['time']:<10.2f}")