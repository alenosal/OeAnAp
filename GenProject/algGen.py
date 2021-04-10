import numpy as np
from pip._vendor.msgpack.fallback import xrange

def nbits(a, b, dx):
    B = (int)((b - a) / dx + 1).bit_length()
    dx_new = (b - a) / (2 ** B - 1)
    return B, dx_new


def gen_population(P, N, B):
    pop = np.random.randint(2, size=(P, N * B))
    return pop


def decode_individual(individual, N, B, a, dx):
    decoded = np.ones(N)
    for i in range(N):
        bits = ''
        for j in range(B):
            bits = bits + str(individual[i * B + j])
        decoded[i] = a + int(bits, 2) * dx
    return decoded


def evaluate_population(f, pop, N, B, a, dx):
    evaluated = np.ones(pop.shape[0])

    for i in range(pop.shape[0]):
        x = decode_individual(pop[i], N, B, a, dx)
        evaluated[i] = f(x)
    return evaluated


  def evolve(p, n, b, pop, pk, pm):
    for generation in range(gen_population(p, n, b)):
        selection = cross(pop, pk)

        elite_index = argmax([selection])
        elite_member = selection[elite_index]

    for i in range(len(selection)):
        selection[i] = mutate(selection[i], pm)
    population = list(sorted(selection,
                            key=lambda x: i,
                            reverse=True))
    yield generation

def argmax(values):
  """Returns the index of the largest value in a list."""
  return max(enumerate(values), key=lambda x: x[1])[0]

def get_best(pop, evaluated_pop):
    best_individual = pop[np.argmax(evaluated_pop)]
    best_value = evaluated_pop[np.argmax(evaluated_pop)]
    return best_individual, best_value


def roulette(pop, evaluated_pop):
    min = np.min(evaluated_pop)
    max = evaluated_pop + (np.abs(min) + 1)
    suma = max.sum()
    max = max / suma
    tmp = 0.0
    amount = pop.shape[0]

    for i in range(amount):
        max[i] = max[i] + tmp
        tmp = max[i]

    new_pop = np.zeros((pop.shape), dtype=int)
    for i in range(amount):
        tmp = np.random.rand()
        for j in range(amount):
            if tmp < max[j]:
                new_pop[i] = pop[j]
                break
    return new_pop


def tournament_selection(pop, scores, k):
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def cross(pop, pk):
    numberOfBites = pop.shape[1]
    popNumber = pop.shape[0]
    new_pop = np.zeros((pop.shape), dtype=int)
    rest = popNumber % 2
    if rest == 1:  # jak nieparzyscie to biore o 1 mniej
        popNumber = popNumber - 1
        new_pop[popNumber] = pop[popNumber]

    for i in range(int(popNumber) // 2):  # co 2 żeby nie laczyc juz polaczonych
        tmp = np.random.rand()  # losuje prawdopodobienstwo
        if tmp < pk:
            first_cut = np.random.randint(1, numberOfBites - 1)
            left = pop[2 * i][0:first_cut]  # lewa czesc 1 osobnika
            right = pop[2 * i + 1][first_cut:]  # prawa czesc 2 osobnika
            left_rest = pop[2 * i + 1][0:first_cut]  # lewa czesc 2 osobnika
            right_rest = pop[2 * i][first_cut:]  # prawa czesc pierwszego soobnika
            new_pop[2 * i] = np.append(left, right)  # w miejsce pierwszego łącze pierwsze dwie
            new_pop[2 * i + 1] = np.append(left_rest, right_rest)  # w miejsce drugiego łącze pozostałe dwie
            continue  # nastepny obrot petli
        new_pop[2 * i] = pop[2 * i]  # zostaje
        new_pop[2 * i + 1] = pop[2 * i + 1]  # zostaje
    return new_pop

def crossoverOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    point = np.random.randint(1, size - 1)
    ind1[point:], ind2[point:] = ind2[point:], ind1[point:]

    return ind1, ind2

def crossoverTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    point1 = np.random.randint(1, size)
    point2 = np.random.randint(1, size - 1)
    if point2 >= point1:
        point2 += 1
    else:
        point1, point2 = point2, point1

    ind1[point1:point2], ind2[point1:point2] \
        = ind2[point1:point2], ind1[point1:point2]

    return ind1, ind2

def crossoverUniform(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    for i in xrange(size):
        if np.random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def inv_mutation(chromosomes, mutation_rate):
    mutated_chromosomes = []

    for chromosome in chromosomes:

        if np.random.random() < mutation_rate:
            r1 = np.random.randint(0, len(chromosome) - 1)
            r2 = np.random.randint(0, len(chromosome) - 1)

            if r1 < r2:
                mutated_chromosomes.append(chromosome[:r1] + chromosome[r1:r2][::-1] + chromosome[r2:])
            else:
                mutated_chromosomes.append(chromosome[:r2] + chromosome[r2:r1][::-1] + chromosome[r1:])

        else:
            mutated_chromosomes.append(chromosome)

    return mutated_chromosomes

def two_point_cross(pop, pk):
    number_of_bites = pop.shape[1]
    pop_number = pop.shape[0]
    new_pop = np.zeros((pop.shape), dtype=int)
    rest = pop_number % 2
    if rest == 1:  # jak nieparzyscie to biore o 1 mniej
        pop_number = pop_number - 1
        new_pop[pop_number] = pop[pop_number]

    for i in range(int(pop_number) // 2):
        tmp = np.random.rand()  # losuje prawdopodobienstwo
        if tmp < pk:
            first_cut = np.random.randint(1, number_of_bites // 2)
            second_cut = np.random.randint(first_cut + 1, number_of_bites - 1)

            left = pop[2 * i][0:first_cut]
            middle = pop[2 * i + 1][first_cut, second_cut]
            right = pop[2 * i][second_cut:]

            left_rest = pop[2 * i + 1][0:first_cut]
            middle_rest = pop[2 * i][first_cut:second_cut]
            right_rest = pop[2 * i + 1][second_cut:]

            new_pop[2 * i] = np.append(left, middle, right)  # w miejsce pierwszego łącze pierwsze dwie
            new_pop[2 * i + 1] = np.append(left_rest, middle_rest, right_rest)
            continue
        new_pop[2 * i] = pop[2 * i]  # zostaje
        new_pop[2 * i + 1] = pop[2 * i + 1]  # zostaje
    return new_pop


def three_point_cross(pop, pk):
    number_of_bites = pop.shape[1]
    pop_number = pop.shape[0]
    new_pop = np.zeros((pop.shape), dtype=int)
    rest = pop_number % 2
    if rest == 1:  # jak nieparzyscie to biore o 1 mniej
        pop_number = pop_number - 1
        new_pop[pop_number] = pop[pop_number]

    for i in range(int(pop_number) // 2):
        tmp = np.random.rand()  # losuje prawdopodobienstwo
        if tmp < pk:
            first_cut = np.random.randint(1, number_of_bites // 3)
            second_cut = np.random.randint(first_cut + 1, (number_of_bites // 3) * 2)
            third_cut = np.random.randint(second_cut, number_of_bites - 1)

            left = pop[2 * i][0:first_cut]
            left_mid = pop[2 * i + 1][first_cut:second_cut]
            right_mid = pop[2 * i][second_cut:third_cut]
            right = pop[2 * i + 1][third_cut:]

            left_rest = pop[2 * i + 1][0:first_cut]
            left_mid_rest = pop[2 * i][first_cut:second_cut]
            right_mid_rest = pop[2 * i + 1][second_cut:third_cut]
            right_rest = pop[2 * i][third_cut:]

            new_pop[2 * i] = np.append(left, left_mid, right_mid, right)  # w miejsce pierwszego łącze pierwsze dwie
            new_pop[2 * i + 1] = np.append(left_rest, left_mid_rest, right_mid_rest, right_rest)
            continue
        new_pop[2 * i] = pop[2 * i]  # zostaje
        new_pop[2 * i + 1] = pop[2 * i + 1]  # zostaje
        return new_pop


def mutate(pop, pm):
    new_pop = pop.copy()

    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            prob = np.random.rand()
            if prob < pm:
                if new_pop[i][j] == 0:
                    new_pop[i][j] = 1
                else:
                    new_pop[i][j] = 0

    return new_pop


def two_point_mutation(pop, pm):
    new_pop = pop.copy()
    for i in range(pop.shape[0]):
        first_bit = np.random.randint(0, pop.shape[1])
        second_bit = np.random.randint(0, pop.shape[1])
        while second_bit == first_bit:
            first_bit = np.random.randint(0, pop.shape[1])
            second_bit = np.random.randint(0, pop.shape[1])
        prob = np.random.rand()
        if prob < pm:
            if new_pop[i][first_bit] == 0:
                new_pop[i][first_bit] = 1
            else:
                new_pop[i][first_bit] = 0

            if new_pop[i][second_bit] == 0:
                new_pop[i][second_bit] = 1
            else:
                new_pop[i][second_bit] = 0
    return new_pop


def inversion(pop, pk):
    new_pop = pop.copy()
    number_of_bites = pop.shape[1]
    pop_number = pop.shape[0]
    for i in range(pop_number):
        prob = np.random.rand()
        if prob < pk:
            first_cut = np.random.randint(1, number_of_bites // 2)
            second_cut = np.random.randint(first_cut + 1, number_of_bites - 1)
            tmp = pop[i][first_cut:second_cut].copy()
            tmp.reverse()
            new_pop[i][first_cut:second_cut] = tmp.copy()
    return new_pop


first_best = None
middle_best = None
last_best = None



