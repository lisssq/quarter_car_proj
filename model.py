import numpy as np

class QuarterCarModel:
    
    def __init__(self, m1=40, m2=850, k1=200000):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        
        self.t_end = 5.0      # время моделирования
        self.dt = 0.001       # шаг по времени, с
        self.t = np.arange(0, self.t_end, self.dt)
        
        self.z0 = np.zeros_like(self.t) # дорожное возмущение
        self.z0[self.t >= 0.5] = 0.05
        
        self.w1 = 0.7  # вес комфорта (ускорение кузова)
        self.w2 = 0.2  # вес управляемости (деформация шины)
        self.w3 = 0.1  # вес хода подвески
    
    def equations_of_motion(self, y, t, k2, c2, z0_val):
        z1, z1_dot, z2, z2_dot = y
        
        F_susp = k2 * (z2 - z1) + c2 * (z2_dot - z1_dot)
        F_tire = self.k1 * (z1 - z0_val)
        
        dz1_dt = z1_dot
        dz1_dot_dt = (F_susp - F_tire) / self.m1
        dz2_dt = z2_dot
        dz2_dot_dt = -F_susp / self.m2
        
        return np.array([dz1_dt, dz1_dot_dt, dz2_dt, dz2_dot_dt])
    
    def integrate(self, k2, c2):
        n = len(self.t)
        
        z1 = np.zeros(n)
        z1_dot = np.zeros(n)
        z2 = np.zeros(n)
        z2_dot = np.zeros(n)
        
        z1[0] = z1_dot[0] = z2[0] = z2_dot[0] = 0
        
        for i in range(n-1):    # интегрирование 
            y_current = np.array([z1[i], z1_dot[i], z2[i], z2_dot[i]])
            
            k1 = self.dt * self.equations_of_motion(
                y_current, self.t[i], k2, c2, self.z0[i])
            
            k2_rk = self.dt * self.equations_of_motion(
                y_current + 0.5*k1, self.t[i] + 0.5*self.dt, k2, c2, self.z0[i])
            
            k3 = self.dt * self.equations_of_motion(
                y_current + 0.5*k2_rk, self.t[i] + 0.5*self.dt, k2, c2, self.z0[i])
            
            k4 = self.dt * self.equations_of_motion(
                y_current + k3, self.t[i] + self.dt, k2, c2, self.z0[i])
            
            y_next = y_current + (k1 + 2*k2_rk + 2*k3 + k4) / 6
            z1[i+1], z1_dot[i+1], z2[i+1], z2_dot[i+1] = y_next
        
        # вычисление ускорения кузова
        z2_ddot = np.zeros(n)
        z2_ddot[1:-1] = (z2_dot[2:] - z2_dot[:-2]) / (2*self.dt)
        z2_ddot[0] = z2_ddot[1]
        z2_ddot[-1] = z2_ddot[-2]
        
        return z1, z2, z2_ddot
    
    def evaluate(self, k2, c2):
        z1, z2, z2_ddot = self.integrate(k2, c2)
        
        tire_deflection = z1 - self.z0      # деформация шины
        suspension_travel = z2 - z1         # ход подвески
        
        J_comfort = np.sqrt(np.mean(z2_ddot**2))           # комфорт
        J_handling = np.sqrt(np.mean(tire_deflection**2))  # управляемость
        J_travel = np.sqrt(np.mean(suspension_travel**2))  # ход подвески
        
        J_total = self.w1 * J_comfort + self.w2 * J_handling + self.w3 * J_travel
        
        return {
            'k2': k2,
            'c2': c2,
            'J_total': J_total,
            'J_comfort': J_comfort,
            'J_handling': J_handling,
            'J_travel': J_travel,
            'z1': z1,
            'z2': z2,
            'z2_ddot': z2_ddot,
            'tire_deflection': tire_deflection,
            'suspension_travel': suspension_travel
        }
    
    def objective_function(self, x):
        k2, c2 = x
        
        if k2 < 15000 or k2 > 50000 or c2 < 1000 or c2 > 8000:
            return 1e10  
        
        result = self.evaluate(k2, c2)
        return result['J_total']