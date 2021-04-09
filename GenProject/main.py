import math
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from algGen import *
import time

def mccormickFunction(x):
    return math.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def functionPlot():
    samples = 500
    sampled = np.linspace(5, -5, samples).astype(int)
    x, y = np.meshgrid(sampled, sampled)
    z = np.zeros((len(sampled), len(sampled)))
    for i in range(len(sampled)):
        for j in range(len(sampled)):
            z[i, j] = mccormickFunction(np.array([x[i][j], y[i][j]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    ax.view_init(10, 50)
    plt.show()

functionPlot()


def algorithm(fun, pop_size, pk, generations, dx, flag):
    N = 2
    B, dx_new = nbits(-10, 10, dx)
    k = 3
    pop = gen_population(pop_size, N, B)
    evaluated_pop = evaluate_population(fun, pop, N, B, -10, dx_new)
    best_sol, best_val = get_best(pop, evaluated_pop)
    global first_best  #
    first_best = best_sol

    best_generation = 0
    list_best = [best_val]
    list_best_generation = [best_val]
    mean = float(sum(evaluated_pop)) / float(pop_size)
    list_mean = [mean]
    start_time = time.time()

    for i in range(1, generations):
        scores = [c for c in pop]
        selected = [tournament_selection(pop, scores, k) for _ in range(generations)]
        pop = selected[i]
        #pk = selected[i+1] dlaczego podstawiamy pk?
        if flag == "roulette":
            pop = roulette(pop, evaluated_pop)
        if flag == "cross":
            pop = cross(pop, pk)
        if flag == "two_point_cross":
            pop = two_point_cross(pop, pk)
        if flag == "three_point_cross":
            pop = three_point_cross(pop, pk)
        if flag == "mutate":
            pop = mutate(pop, pk)
        if flag == "two_point_mutation":
            pop = two_point_mutation(pop, pk)
        if flag == "inversion":
            pop = inversion(pop, pk)

        evaluated_pop = evaluate_population(fun, pop, N, B, -10, dx_new)
        best_candidate, best_val_candidate = get_best(pop, evaluated_pop)

        if i == 100:
            global middle_best
            middle_best = best_candidate

        if i == 199:
            global last_best
            last_best = best_candidate

        if best_val_candidate > list_best[i - 1]:
            best_sol = best_candidate
            best_generation = i
            list_best.append(best_val_candidate)
        else:
            list_best.append(list_best[i - 1])

        list_best_generation.append(best_val_candidate)
        list_mean.append(float(sum(evaluated_pop)) / float(pop_size))

        execution_time = start_time - time.time()

    return best_sol, best_generation, list_best, list_best_generation, list_mean, execution_time
