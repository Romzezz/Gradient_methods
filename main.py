#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sp
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from scipy.optimize import line_search, fmin
from one_dim_optimization import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time


# In[2]:


def _gradient(expr, point):
    grad = []
    symbols = list(expr.free_symbols)
    sub = dict(zip(symbols, point))
    for i in range(len(point)):
        curr_symbol = symbols[i]
        grad.append(expr.diff(curr_symbol).subs(sub))
    return np.array(grad, dtype=float)


# In[3]:


def _visualize(func, history):
    x_points = [point[0] for point in history['x']]
    y_points = [point[1] for point in history['x']]
    x_min, x_max = min(x_points), max(x_points)
    y_min, y_max = min(y_points), max(y_points)
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    scatter = data=go.Scatter3d(
        x=x_points, y=y_points, z=history['f'], mode='lines+markers', marker={'size':4})
    
    surface = go.Surface(z=Z, x=X, y=Y, opacity=0.5, colorscale='inferno')
    
    fig = go.Figure(data=[surface, scatter])
    
    fig.update_layout(title='График функции')
    fig.show()


# In[4]:


def ask_input(ask_alpha=False, ask_alpha0=False, ask_delta=False, ask_gamma=False, ask_history=False, ask_visualizing=False):
    result = dict()
    func = input('Введите функцию в аналитическом виде: ')
    result['func'] = func
    x0 = list(map(float, input('Введите начальную точку: ').split()))
    result['x0'] = x0
    
    if ask_alpha:
        alpha = input('Введите константный шаг (оставьте пустым для значения по умолчанию):')
        alpha = 0.1 if not alpha else float(alpha)
        result['alpha'] = alpha
        
    if ask_alpha0:
        alpha0 = input('Введите начальный шаг (оставьте пустым для значения по умолчанию):')
        alpha0 = 0.1 if not alpha0 else float(alpha0)
        result['alpha0'] = alpha0
        
    if ask_delta:
        delta = input('Введите значение параметра оценки (оставьте пустым для значения по умолчанию): ')
        delta = 0.1 if not delta else float(delta)
        result['delta'] = delta
        
    if ask_gamma:
        gamma = input('Введите значение параметра дробления (оставьте пустым для значения по умолчанию): ')
        gamma = 0.1 if not gamma else float(gamma)
        result['gamma'] = gamma
        
    epsilon = input('Введите точность оптимизации (оставьте пустым для значения по умолчанию): ')
    epsilon = 1e-5 if not epsilon else float(epsilon)
    result['epsilon'] = epsilon
    max_iter = input('Введите максимальное количество итераций (оставьте пустым для значения по умолчанию): ')
    max_iter = 500 if not max_iter else int(max_iter)
    result['max_iter'] = max_iter
    
    if ask_history:
        show_history = bool(int(input('Показать промежуточные результаты? 1-да/0-нет: ')))
        result['show_history'] = show_history
        
    if ask_visualizing:
        visualize = bool(int(input('Визуализировать результат? 1-да/0-нет: ')))
        result['visualize'] = visualize
        
    return result


# In[12]:


def compare():
    data = ask_input(1, 1, 1, 1)
    values = [['Полученное решение', 'Время выполнения (ms)', 'Количество итераций']]
    i_to_algo = {1: constant_gradient_descent, 2: step_splitting_gd,
                 3: fastest_gd, 4: conjugate_gradient_method}
    accuracy = -int(np.log10(data['epsilon']))
    
    for i in range(1, 5):
        algo = i_to_algo[i]
        if i == 1:
            start = time.time()
            res, history = algo(data['func'], data['x0'], data['alpha'], data['max_iter'],
                       data['epsilon'])
            end = time.time()
            duration = round(end - start, accuracy)
            solution = res['point']
            iter_num = len(history)
            values.append([solution, duration, iter_num])
            
        elif i == 2:
            start = time.time()
            res, history = algo(data['func'], data['x0'], data['alpha0'], data['delta'], data['gamma'],
                       data['max_iter'], data['epsilon'])
            end = time.time()
            duration = round(end - start, accuracy)
            solution = res['point']
            iter_num = len(history)
            values.append([solution, duration, iter_num])
            
        else:
            start = time.time()
            res, history = algo(data['func'], data['x0'], data['max_iter'],
                       data['epsilon'])
            end = time.time()
            duration = round(end - start, accuracy)
            solution = res['point']
            iter_num = len(history)
            values.append([solution, duration, iter_num])
    
    expr = sp.sympify(data['func'])
    symbols = expr.free_symbols
    lambdifyed = sp.lambdify(symbols, expr)
    acc_time_start = time.time()
    res = fmin(lambda _x: lambdifyed(*_x), data['x0'])
    acc_time_end = time.time()
    acc_duration = round(acc_time_end - acc_time_start, accuracy)
    solution = [round(x, accuracy) for x in res]
    values.append([solution, acc_duration, '-'])
    fig = go.Figure(data=[go.Table(header=dict(values=['Параметр', 'константный шаг', 'дробление шага', 'наискорейший спуск', 'Ньютон-сопряженный градиент', 'Оптимальный точный алгоритм']),
                                   cells=dict(values=values))])
    fig.show()


# In[6]:


def gradient_descent():
    method = int(input(
        """
        Выберите метод решения:
        1 - градиентный спуск с постоянным шагом
        2 - градиентный спуск с дроблением шага
        3 - наискорейший градиентный спуск
        4 - алгоритм Ньютона-сопряженного градиента
        Метод: 
        """
    ))
    if method == 1:
        data = ask_input(ask_alpha=True, ask_history=True, ask_visualizing=True)
        return constant_gradient_descent(data['func'], data['x0'], data['alpha'], data['max_iter'],
                                         data['epsilon'], data['show_history'], data['visualize'])[0]
    elif method == 2:
        data = ask_input(ask_alpha0=True, ask_delta=True, ask_gamma=True, ask_history=True, ask_visualizing=True)
        return step_splitting_gd(data['func'], data['x0'], data['alpha0'], data['delta'], data['gamma'],
                              data['max_iter'], data['epsilon'], data['show_history'], data['visualize'])[0]
    elif method == 3:
        data = ask_input(ask_history=True, ask_visualizing=True)
        return fastest_gd(data['func'], data['x0'], data['max_iter'],
                                         data['epsilon'], data['show_history'], data['visualize'])[0]
    elif method == 4:
        data = ask_input(ask_history=True, ask_visualizing=True)
        return conjugate_gradient_method(data['func'], data['x0'], data['max_iter'],
                                         data['epsilon'], data['show_history'], data['visualize'])[0]
    


# In[7]:


def constant_gradient_descent(func, x0, alpha=0.1, max_iter=500, epsilon=1e-5, show_history=False, visualize=False):
    x = np.array(x0, dtype=float)
    expr = sp.sympify(func)
    symbols = expr.free_symbols
    lambdifyed = sp.lambdify(symbols, expr)
    grad = _gradient(expr, x)
    f_x = lambdifyed(*x)
    history = pd.DataFrame({'Iter': [0], 'x': [x], 'f': f_x, '||grad||': [np.sum(grad**2)**0.5]})
    accuracy = int(np.log10(1/epsilon))
    
    for i in range(1, max_iter):
        if np.sum(grad**2)**0.5 < epsilon:
            history['code'] = 0
            break
        else:
            x = x - alpha * grad
            grad = _gradient(expr, x)
            f_x = lambdifyed(*x)

        row = {'Iter': i, 'x': x, 'f': f_x, '||grad||': np.sum(grad**2)**0.5}
        history = history.append(row, ignore_index=True)

    else:
        history['code'] = 1
        
    if show_history:
        for column in history.columns:
            if column != 'x':
                history[column] = [round(value, accuracy) for value in history[column]]
            else:
                history[column] = [[round(value, accuracy) for value in arr] for arr in history[column]]
        history.set_index('Iter', inplace=True)
        display(HTML(history.to_html()))
    
    if visualize:
        if len(x0) == 2:
            _visualize(lambdifyed, history)

    return {'point': np.array([round(val, accuracy) for val in x]), 'f': round(f_x, accuracy)}, history


# In[8]:


def step_splitting_gd(func, x0, alpha0=0.1, delta=0.1, gamma=0.1, max_iter=500, epsilon=1e-5, show_history=False, visualize=False):
    x = np.array(x0, dtype=float)
    expr = sp.sympify(func)
    symbols = expr.free_symbols
    lambdifyed = sp.lambdify(symbols, expr)
    grad = _gradient(expr, x)
    f_x = lambdifyed(*x)
    history = pd.DataFrame({'Iter': [0], 'x': [x], 'f': f_x, '||grad||': [np.sum(grad**2)**0.5]})
    accuracy = int(np.log10(1/epsilon))
    
    try:
        for i in range(1, max_iter):
            
            t = x - alpha0 * grad
            f_t = lambdifyed(*t)

            while not f_t - f_x <= - delta * alpha0 * sum(grad ** 2):
                alpha0 *= gamma
                t = x - gamma * grad
                f_t = lambdifyed(*t)

            x = t
            f_x = f_t
            grad = _gradient(expr, x)

            row = {'Iter': i, 'x': x, 'f': f_x, '||grad||': np.sum(grad**2)**0.5}
            history = history.append(row, ignore_index=True)
            
            if np.sum(grad**2)**0.5 < epsilon:
                history['code'] = 0
                break

        else:
            history['code'] = 1
            
    except Exception as e:
        history['code'] = 2
        
    if show_history:
        for column in history.columns:
            if column != 'x':
                history[column] = [round(value, accuracy) for value in history[column]]
            else:
                history[column] = [[round(value, accuracy) for value in arr] for arr in history[column]]
        history.set_index('Iter', inplace=True)
        display(HTML(history.to_html()))
        
    if visualize:
        if len(x0) == 2:
            _visualize(lambdifyed, history)

    return {'point': np.array([round(val, accuracy) for val in x]), 'f': round(f_x, accuracy)}, history


# In[9]:


def fastest_gd(func, x0, max_iter=500, epsilon=1e-5, show_history=False, visualize=False):
    x = np.array(x0, dtype=float)
    expr = sp.sympify(func)
    symbols = expr.free_symbols
    lambdifyed = sp.lambdify(symbols, expr)
    grad = _gradient(expr, x)
    f_x = lambdifyed(*x)
    history = pd.DataFrame({'Iter': [0], 'x': [x], 'f': f_x, '||grad||': [np.sum(grad**2)**0.5]})
    accuracy = int(np.log10(1/epsilon))
    
    try:
        for i in range(1, max_iter):
            alpha = combined_brent(lambda lr: lambdifyed(*(x - lr*grad)), (0, 1))['arg']
            x = x - alpha * grad
            grad = _gradient(expr, x)
            f_x = lambdifyed(*x)

            row = {'Iter': i, 'x': x, 'f': f_x, '||grad||': np.sum(grad**2)**0.5}
            history = history.append(row, ignore_index=True)

            if np.sum(grad**2)**0.5 < epsilon:
                history['code'] = 0
                break

        else:
            history['code'] = 1
    
    except Exception as e:
        history['code'] = 2
        
    if show_history:
        for column in history.columns:
            if column != 'x':
                history[column] = [round(value, accuracy) for value in history[column]]
            else:
                history[column] = [[round(value, accuracy) for value in arr] for arr in history[column]]
        history.set_index('Iter', inplace=True)
        display(HTML(history.to_html()))

    if visualize:
        if len(x0) == 2:
            _visualize(lambdifyed, history)

    return {'point': np.array([round(val, accuracy) for val in x]), 'f': round(f_x, accuracy)}, history


# In[10]:


def conjugate_gradient_method(func, x0, max_iter=500, epsilon=1e-5, show_history=False, visualize=False):
    x = np.array(x0, dtype=float)
    expr = sp.sympify(func)
    symbols = expr.free_symbols
    lambdifyed = sp.lambdify(symbols, expr)
    vectorized = lambda _x: lambdifyed(*x)
    grad = _gradient(expr, x)
    p = grad
    f_x = lambdifyed(*x)
    history = pd.DataFrame({'Iter': [0], 'x': [x], 'f': f_x, '||grad||': [np.sum(grad**2)**0.5]})
    accuracy = int(np.log10(1/epsilon)) 
    
    for i in range(1, max_iter):
        if np.sum(grad**2)**0.5 < epsilon:
            history['code'] = 0
            break
        else:
            alpha = line_search(vectorized, lambda _x: _gradient(expr, _x), x, p)[0]
            
            if alpha is None:
                alpha = combined_brent(lambda lr: lambdifyed(*(x - lr*p)), (0, 1))['arg']
                    
            x = x - alpha * p
            grad_new = _gradient(expr, x)
            beta_fr = (grad_new @ grad_new.reshape(-1, 1)) / (grad @ grad.reshape(-1, 1))
            p = grad_new + beta_fr * p
            grad = grad_new
            f_x = lambdifyed(*x)

        row = {'Iter': i, 'x': x, 'f': f_x, '||grad||': np.sum(grad**2)**0.5}
        history = history.append(row, ignore_index=True)

    else:
        history['code'] = 1
        
    if show_history:
        for column in history.columns:
            if column != 'x':
                history[column] = [round(value, accuracy) for value in history[column]]
            else:
                history[column] = [[round(value, accuracy) for value in arr] for arr in history[column]]
        history.set_index('Iter', inplace=True)
        display(HTML(history.to_html()))
        
    if visualize:
        if len(x0) == 2:
            _visualize(lambdifyed, history)

    return {'point': np.array([round(val, accuracy) for val in x]), 'f': round(f_x, accuracy)}, history


# In[16]:


gradient_descent()


# In[13]:


compare()


# In[ ]:



