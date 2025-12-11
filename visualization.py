import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_influence(model):
    print("\n" + "="*60)
    print("ВЛИЯНИЕ ПАРАМЕТРОВ ПОДВЕСКИ НА ПОВЕДЕНИЕ АВТОМОБИЛЯ")
    print("="*60)
    
    configurations = {
        'Мягкая (комфорт)': {'k2': 18000, 'c2': 2000},
        'Средняя (универсал)': {'k2': 30000, 'c2': 4000},
        'Жесткая (спорт)': {'k2': 45000, 'c2': 6000},
        'Слабое демпфирование': {'k2': 30000, 'c2': 1200},
        'Сильное демпфирование': {'k2': 30000, 'c2': 7000}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Влияние параметров подвески на поведение автомобиля', fontsize=14)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(configurations)))
    
    for idx, (name, params) in enumerate(configurations.items()):
        k2 = params['k2']
        c2 = params['c2']
        
        result = model.evaluate(k2, c2)
        
        # 1. ускорение кузова (комфорт)
        axes[0, 0].plot(model.t, result['z2_ddot'], 
                       label=f'{name}\nk2={k2:.0f}, c2={c2:.0f}', 
                       color=colors[idx], alpha=0.7, linewidth=1.5)
        
        # 2. деформация шины (управляемость)
        axes[0, 1].plot(model.t, result['tire_deflection'], 
                       color=colors[idx], alpha=0.7, linewidth=1.5)
        
        # 3. ход подвески
        axes[1, 0].plot(model.t, result['suspension_travel'], 
                       color=colors[idx], alpha=0.7, linewidth=1.5)
        
        # 4. положение кузова
        axes[1, 1].plot(model.t, result['z2'], 
                       label=name, color=colors[idx], alpha=0.7, linewidth=1.5)
        
        print(f"\n{name}:")
        print(f"  k2 = {k2:.0f} Н/м, c2 = {c2:.0f} Н·с/м")
        print(f"  Комфорт (ускорение): {result['J_comfort']:.4f} м/с²")
        print(f"  Управляемость (деформация шины): {result['J_handling']:.6f} м")
        print(f"  Ход подвески: {result['J_travel']:.6f} м")
        print(f"  Общий критерий J: {result['J_total']:.4f}")
    
    axes[0, 0].set_title('Ускорение кузова (Комфорт)')
    axes[0, 0].set_ylabel('м/с²')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    axes[0, 0].set_xlim(0.4, 2.0)
    
    axes[0, 1].set_title('Деформация шины (Управляемость)')
    axes[0, 1].set_ylabel('м')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.4, 2.0)
    
    axes[1, 0].set_title('Ход подвески')
    axes[1, 0].set_xlabel('Время, с')
    axes[1, 0].set_ylabel('м')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0.4, 2.0)
    
    axes[1, 1].set_title('Положение кузова')
    axes[1, 1].set_xlabel('Время, с')
    axes[1, 1].set_ylabel('м')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc='upper right')
    axes[1, 1].set_xlim(0.4, 2.0)
    
    plt.tight_layout()
    plt.show()

def plot_optimal_solutions(model, optimization_results):
    print("\n" + "="*60)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение оптимальных решений разных методов', fontsize=14)
    
    colors = {'Nelder-Mead': 'blue', 
              'Hyperopt': 'orange', 
              'Optuna': 'green',}
    
    detailed_results = []
    
    for res in optimization_results:
        method = res['method']
        k2 = res['k2']
        c2 = res['c2']
        
        result = model.evaluate(k2, c2)
        
        label = f"{method}\nk2={k2:.0f}, c2={c2:.0f}"
        
        # 1. ускорение кузова
        axes[0, 0].plot(model.t, result['z2_ddot'], 
                       label=label, 
                       color=colors.get(method, 'black'), 
                       alpha=0.8, linewidth=1.5)
        
        # 2. деформация шины
        axes[0, 1].plot(model.t, result['tire_deflection'], 
                       color=colors.get(method, 'black'), 
                       alpha=0.8, linewidth=1.5)
        
        # 3. ход подвески
        axes[1, 0].plot(model.t, result['suspension_travel'], 
                       color=colors.get(method, 'black'), 
                       alpha=0.8, linewidth=1.5)
        
        # 4. положение кузова
        axes[1, 1].plot(model.t, result['z2'], 
                       color=colors.get(method, 'black'), 
                       alpha=0.8, linewidth=1.5)
        
        detailed_results.append({
            'method': method,
            'k2': k2,
            'c2': c2,
            'J_total': result['J_total'],
            'J_comfort': result['J_comfort'],
            'J_handling': result['J_handling'],
            'J_travel': result['J_travel']
        })
    
    axes[0, 0].set_title('Ускорение кузова (Комфорт)')
    axes[0, 0].set_ylabel('м/с²')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc='upper right')
    axes[0, 0].set_xlim(0.4, 2.0)
    
    axes[0, 1].set_title('Деформация шины (Управляемость)')
    axes[0, 1].set_ylabel('м')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.4, 2.0)
    
    axes[1, 0].set_title('Ход подвески')
    axes[1, 0].set_xlabel('Время, с')
    axes[1, 0].set_ylabel('м')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0.4, 2.0)
    
    axes[1, 1].set_title('Положение кузова')
    axes[1, 1].set_xlabel('Время, с')
    axes[1, 1].set_ylabel('м')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0.4, 2.0)
    
    plt.tight_layout()
    plt.show()
    
    _print_detailed_comparison(detailed_results)

def _print_detailed_comparison(results):
    print("\n" + "="*60)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ ОПТИМАЛЬНЫХ РЕШЕНИЙ")
    
    print(f"\n{'Метод':<25} {'k2':<8} {'c2':<8} {'J общ':<10} {'J комф':<10} {'J упр':<10} {'J ход':<10}")
    print("-" * 85)
    
    for res in results:
        print(f"{res['method']:<25} "
              f"{res['k2']:<8.0f} "
              f"{res['c2']:<8.0f} "
              f"{res['J_total']:<10.4f} "
              f"{res['J_comfort']:<10.4f} "
              f"{res['J_handling']:<10.6f} "
              f"{res['J_travel']:<10.6f}")

def plot_optimizer_comparison(optimization_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = [r['method'] for r in optimization_results]
    J_values = [r['J'] for r in optimization_results]
    times = [r['time'] for r in optimization_results]
    iterations = [r['iterations'] for r in optimization_results]
    
    colors = ['blue', 'orange', 'green']
    
    # 1. сравнение значений целевой функции
    bars1 = axes[0].bar(range(len(methods)), J_values, color=colors)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=0)
    axes[0].set_ylabel('Значение J')
    axes[0].set_title('Качество решения (меньше = лучше)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, J_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. сравнение времени выполнения
    bars2 = axes[1].bar(range(len(methods)), times, color=colors)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=0)
    axes[1].set_ylabel('Время, с')
    axes[1].set_title('Время выполнения')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f} с', ha='center', va='bottom', fontsize=9)
    
    # 3. сравнение количества итераций
    bars3 = axes[2].bar(range(len(methods)), iterations, color=colors)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=0)
    axes[2].set_ylabel('Итерации')
    axes[2].set_title('Количество итераций')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, iterations):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()