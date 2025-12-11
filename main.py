import warnings
warnings.filterwarnings('ignore')

from model import QuarterCarModel
from optimizers import Optimizer
from visualization import plot_parameter_influence, plot_optimal_solutions, plot_optimizer_comparison

def main():
    print("\n[1] СОЗДАНИЕ МАТЕМАТИЧЕСКОЙ МОДЕЛИ")
    model = QuarterCarModel()
    print(f"Параметры модели:")
    print(f"  m1 = {model.m1} кг (неподрессоренная масса)")
    print(f"  m2 = {model.m2} кг (подрессоренная масса)")
    print(f"  k1 = {model.k1} Н/м (жесткость шины)")
    print(f"  Веса критериев: w1={model.w1} (комфорт), w2={model.w2} (управляемость), w3={model.w3} (ход подвески)")
    
    print("\n[2] ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ ПОДВЕСКИ")
    print("Изучаем 5 типов подвески: 1) мягкая, 2) средняя, 3) жесткая, 4) со слабым и 5) сильным демпфированием...")
    plot_parameter_influence(model)
    
    print("\n[3] СРАВНЕНИЕ МЕТОДОВ ОПТИМИЗАЦИИ")
    print("Запускаем 3 оптимизатора: Nelder-Mead, Hyperopt, Optuna...")
    
    optimizer = Optimizer(model)
    optimization_results = optimizer.compare_all()
    
    plot_optimizer_comparison(optimization_results)
    
    print("\n[4] ВИЗУАЛИЗАЦИЯ ОПТИМАЛЬНЫХ РЕШЕНИЙ")
    print("Сравниваем поведение системы при параметрах, найденных разными методами оптимизации...")
    plot_optimal_solutions(model, optimization_results)
    
    print("\n" + "=" * 70)
    print("УСПЕШНО ЗАВЕРШЕНО")
    print("=" * 70)

if __name__ == "__main__":
    main()