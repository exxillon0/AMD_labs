import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

x1, x2 = sp.symbols('x1 x2', real=True)
f_sym = (x1 + x2)**6 - 2*(x1 + x2) - 7

x0 = (0.5, 0.5)
# new
# сетка
x1_vals = np.linspace(0, 1, 20)
x2_vals = np.linspace(0, 1, 20)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Численно вычисляем ф-цию
f_lambdified = sp.lambdify((x1, x2), f_sym, modules='numpy')
f_original = f_lambdified(X1, X2)

# многомерный ряд тейлора
def taylor_multivariate(f, vars, point, order):
    h = [sp.Symbol(f'h{i}') for i in range(len(vars))]
    subs_forward = {vars[i]: point[i] + h[i] for i in range(len(vars))}
    f_shift = f.subs(subs_forward)
    
    for i in range(len(vars)):
        f_shift = f_shift.series(h[i], 0, order).removeO()
    
    f_expanded = sp.expand(f_shift)
    if f_expanded.is_Add:
        terms = f_expanded.args
    else:
        terms = [f_expanded]

    filtered = []
    for term in terms:
        if term == 0:
            continue
        if not term.has_free(*h):
            deg = 0
        else:
            poly = sp.Poly(term, *h)
            deg = poly.total_degree()
        if deg < order:
            filtered.append(term)
    
    if filtered:
        f_filtered = sp.Add(*filtered)
    else:
        f_filtered = sp.Integer(0)
    
    #  h_i -> vars_i - point_i
    subs_back = {h[i]: vars[i] - point[i] for i in range(len(vars))}
    result = f_filtered.subs(subs_back)
    return result

# графики
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-60)
ax.grid(True)

# Исходная функция
surf_orig = ax.plot_surface(X1, X2, f_original, alpha=0.5, cmap='jet',
                            edgecolor='none', label='Исходная функция')

orders = [1, 2, 3, 4]
colors = ['red', 'green', 'magenta', 'cyan']

for i, order in enumerate(orders):
    # Символьное выражение ряда Тейлора
    taylor_expr = taylor_multivariate(f_sym, (x1, x2), x0, order)
    
    # Численная функция
    taylor_lambd = sp.lambdify((x1, x2), taylor_expr, modules='numpy')
    f_approx = taylor_lambd(X1, X2)
    
    if np.isscalar(f_approx):
        # Если скаляр, создаем массив такой же формы, как X1
        f_approx = np.full_like(X1, f_approx)
    
    # поверхность
    ax.plot_surface(X1, X2, f_approx, alpha=0.3,
                    edgecolor=colors[i], linewidth=0.5,
                    label=f'Тейлор {order}-го порядка')
    
    # Ошибка
    error = np.mean(np.abs(f_original.flatten() - f_approx.flatten()))
    print(f'Порядок {order}: Средняя ошибка = {error:.4f}')

# Точка разложения
f0 = f_lambdified(x0[0], x0[1])
ax.scatter([x0[0]], [x0[1]], [f0], color='yellow', s=80,
           edgecolor='black', linewidth=1.5, label='Точка разложения')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Аппроксимация функции двух переменных рядом Тейлора')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Исходная функция')
]
for color, order in zip(colors, orders):
    legend_elements.append(
        Line2D([0], [0], color=color, lw=2, label=f'Тейлор {order}-го порядка')
    )
legend_elements.append(
    Line2D([0], [0], marker='o', color='yellow', markersize=10,
           markerfacecolor='yellow', markeredgecolor='black',
           linestyle='None', label='Точка разложения')
)
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.show()