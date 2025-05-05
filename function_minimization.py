import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.optimize import minimize_scalar

class FunctionOptimizer:
    def __init__(self):
        self.methods = {
            "Метод деформируемого многогранника": self.nelder_mead,
            "Градиентный спуск": self.gradient_descent,
            "Сопряжённые градиенты": self.conjugate_gradient,
            "Ньютоновский метод": self.newtonian_method
        }
        
        self.functions = {
            "Розенброка": self.rosenbrock,
            "Химмельблау": self.himmelblau
        }

    @staticmethod
    def rosenbrock(x):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    @staticmethod
    def himmelblau(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def nelder_mead(self, func, start, alpha=1, beta=0.5, gamma=2, eps=1e-3, max_iters=1000):
        n = len(start)
        simplex = [np.array(start)]
        for i in range(n):
            point = np.array(start)
            point[i] += 0.5
            simplex.append(point)
            
        for it in range(max_iters):
            simplex.sort(key=lambda x: func(x))
            if np.std([func(x) for x in simplex]) < eps:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            reflected = centroid + alpha*(centroid - simplex[-1])
            
            if func(reflected) < func(simplex[0]):
                expanded = centroid + gamma*(reflected - centroid)
                simplex[-1] = expanded if func(expanded) < func(reflected) else reflected
            elif func(reflected) < func(simplex[-2]):
                simplex[-1] = reflected
            else:
                contracted = centroid + beta*(simplex[-1] - centroid)
                if func(contracted) < func(simplex[-1]):
                    simplex[-1] = contracted
                else:
                    simplex = [simplex[0] + 0.5*(x - simplex[0]) for x in simplex]
        
        return simplex[0], func(simplex[0]), it

    def gradient(self, func, x, eps=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad

    def gradient_descent(self, func, start, lr=1e-3, eps=1e-4, max_iters=10000):
        x = np.array(start)
        for i in range(max_iters):
            grad = self.gradient(func, x)
            if np.linalg.norm(grad) < eps:
                break
            x -= lr * grad
        return x, func(x), i

    def conjugate_gradient(self, func, start, eps=1e-4, max_iters=10000):
        x = np.array(start)
        grad = self.gradient(func, x)
        d = -grad
        for i in range(max_iters):
            alpha = minimize_scalar(lambda a: func(x + a*d)).x
            x_new = x + alpha*d
            grad_new = self.gradient(func, x_new)
            if np.linalg.norm(grad_new) < eps:
                break
            beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)  # Формула Флетчера-Ривса
            d = -grad_new + beta*d
            x, grad = x_new, grad_new
        return x, func(x), i
    
    def newtonian_method(self, func, start_point, eps=1e-4, max_iter=1000, lambda_reg=1e-2):
        x_k = np.array(start_point, dtype=float)
        iter_count = 0
        
        while iter_count < max_iter:
            grad_k = self.gradient(func, x_k)
            
            if np.linalg.norm(grad_k) < eps:
                break
            
            hess_k = self.hessian(func, x_k)
            
            while True:
                try:
                    hess_reg = hess_k + lambda_reg * np.eye(len(x_k))
                    p_k = -np.linalg.solve(hess_reg, grad_k)
                    break
                except np.linalg.LinAlgError:
                    lambda_reg *= 10
            
            def line_search(alpha):
                return func(x_k + alpha * p_k)
            
            result = minimize_scalar(line_search)
            alpha_k = result.x if result.success else 1.0
            
            x_k += alpha_k * p_k
            iter_count += 1
        
        return x_k, func(x_k), iter_count

    def hessian(self, func, x, eps=1e-5):
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1 = x.copy()
                x1[i] += eps
                x1[j] += eps
                
                x2 = x.copy()
                x2[i] += eps
                x2[j] -= eps
                
                x3 = x.copy()
                x3[i] -= eps
                x3[j] += eps
                
                x4 = x.copy()
                x4[i] -= eps
                x4[j] -= eps
                
                hess[i,j] = (func(x1) - func(x2) - func(x3) + func(x4)) / (4 * eps**2)
        return hess

class OptimizationApp:
    def __init__(self, master):
        self.master = master
        self.optimizer = FunctionOptimizer()
        self.setup_styles()
        self.setup_ui()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))
    
    def setup_ui(self):
        self.master.title("Оптимизатор функций")
        self.master.geometry("1400x900")
        
        control_frame = ttk.Frame(self.master, width=450)
        control_frame.pack(side="left", fill="y", padx=15, pady=15)
        
        ttk.Label(control_frame, text="Целевая функция:").pack(anchor="w", pady=8)
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(control_frame, 
            values=list(self.optimizer.functions.keys()),
            textvariable=self.func_var,
            state="readonly")
        self.func_combobox.pack(fill="x", pady=8)
        self.func_combobox.current(0)
        
        ttk.Label(control_frame, text="Метод оптимизации:").pack(anchor="w", pady=8)
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(control_frame, 
            values=list(self.optimizer.methods.keys()),
            textvariable=self.method_var,
            state="readonly")
        self.method_combobox.pack(fill="x", pady=8)
        self.method_combobox.current(0)
        
        ttk.Label(control_frame, text="Стартовая точка (x1, x2):").pack(anchor="w", pady=8)
        self.x1_entry = ttk.Entry(control_frame)
        self.x1_entry.pack(fill="x", pady=4)
        self.x1_entry.insert(0, "0.0")
        self.x2_entry = ttk.Entry(control_frame)
        self.x2_entry.pack(fill="x", pady=4)
        self.x2_entry.insert(0, "0.0")
        
        ttk.Label(control_frame, text="Точность решения (ε):").pack(anchor="w", pady=8)
        self.eps_entry = ttk.Entry(control_frame)
        self.eps_entry.pack(fill="x", pady=4)
        self.eps_entry.insert(0, "0.0001")
        
        ttk.Button(control_frame, text="Запустить оптимизацию", 
                 command=self.run_optimization).pack(pady=15)
        
        self.figure = plt.figure(figsize=(10,7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=15)
        
        result_frame = ttk.LabelFrame(control_frame, text="Результаты", padding=15)
        result_frame.pack(fill="x", pady=15)
        
        self.result_text = tk.Text(result_frame, height=8, width=40, 
                                 font=('Arial', 12), wrap=tk.WORD)
        self.result_text.pack(fill="x")

    def run_optimization(self):
        try:
            func = self.optimizer.functions[self.func_var.get()]
            method = self.optimizer.methods[self.method_var.get()]
            
            start_point = [
                float(self.x1_entry.get()),
                float(self.x2_entry.get())
            ]
            eps = float(self.eps_entry.get())
            
            result = method(func, start_point, eps=eps)
            self.update_plot(func, result[0])
            self.show_results(result)
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка: {str(e)}")

    def update_plot(self, func, minimum):
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection='3d')
        
        x = np.linspace(minimum[0]-3, minimum[0]+3, 100)
        y = np.linspace(minimum[1]-3, minimum[1]+3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda a,b: func([a,b]))(X, Y)
        
        ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6)
        ax.scatter(*minimum, func(minimum), c='green', s=100, 
                 marker='*', label='Минимум')
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_zlabel('f(X1,X2)', fontsize=12)
        ax.legend()
        self.canvas.draw()

    def show_results(self, result):
        text = f"""{'Оптимальные параметры':^33} 
X₁: {result[0][0]:.8f}
X₂: {result[0][1]:.8f}
Значение функции: {result[1]:.8f}
Итерации: {result[2]}
"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()