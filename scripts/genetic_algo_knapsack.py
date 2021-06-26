import numpy as np  # type: ignore
from typing import List, Tuple


def fitness_function(
    solution: np.array, weights: np.array, values: np.array, max_weight: float
) -> float:
    total_weight = np.dot(solution, weights)
    if total_weight <= max_weight:
        total_value = np.dot(solution, values)
        return total_value
    else:
        return 0


def generate_random_solution(
    random_generator: np.random.Generator, solution_length: int
) -> np.array:
    random_uniforms = random_generator.random(solution_length)
    random_binaries = np.rint(random_uniforms).astype(int)
    return random_binaries


def tournament_select(
    random_generator: np.random.Generator,
    population: List[np.array],
    scores: List[float],
    n: int,
) -> List[np.array]:
    if n % 2 != 0:
        raise ValueError("n must be a multiple of 2")

    tournament_entries = random_generator.choice(
        len(population), size=2 * n, replace=False
    )

    winners = []
    for i in range(n):
        if scores[tournament_entries[2 * i]] > scores[tournament_entries[2 * i + 1]]:
            winners.append(population[tournament_entries[2 * i]])
        else:
            winners.append(population[tournament_entries[2 * i + 1]])
    return winners


def elitist_select(
    population: List[np.array],
    scores: List[float],
    n: int,
) -> List[np.array]:
    if n % 2 != 0:
        raise ValueError("n must be a multiple of 2")

    scores_and_indices = list(enumerate(scores))
    sorted_scores_and_indices = sorted(
        scores_and_indices, key=lambda x: x[1], reverse=True
    )
    return [population[i[0]] for i in sorted_scores_and_indices[0:n]]


# 0.5
def crossover(
    random_generator: np.random.Generator,
    x: np.array,
    y: np.array,
    crossover_prob: float,
) -> Tuple[np.array, np.array]:

    if random_generator.random() < crossover_prob:

        soln_len = len(x)

        a_0 = x[0 : (soln_len // 2)]
        a_1 = y[(soln_len // 2) : soln_len]
        a = np.concatenate([a_0, a_1])
        b = np.concatenate([y[0 : (soln_len // 2)], x[(soln_len // 2) : soln_len]])

        return (a, b)
    else:
        return (x, y)


# 0.05
def mutate(
    random_generator: np.random.Generator, x: np.array, mutation_prob: float
) -> np.array:
    mp = random_generator.random(len(x))
    mp_bool = mp < mutation_prob

    mutated_solution = x.copy()
    mutated_solution[mp_bool] = 1 - mutated_solution[mp_bool]

    return mutated_solution


# consider elitism

if __name__ == "__main__":

    rng = np.random.default_rng(100)

    weights = np.array([7, 2, 1, 9, 3, 7, 6, 12, 4, 1, 1, 1, 3, 8, 23, 77, 2, 8, 3, 44])
    values = np.array(
        [5, 4, 7, 2, 6, 16, 3, 22, 1, 7, 5, 9, 7, 8, 8, 5, 23, 15, 3, 100]
    )
    max_weight = 50
    pop_size = 10
    crossover_prob = 0.5
    mutation_prob = 0.05

    population = [generate_random_solution(rng, 20) for i in range(pop_size)]
    scores = [fitness_function(p, weights, values, max_weight) for p in population]

    print(f"Average Fitness: {np.mean(scores)}")
    print(f"Max Fitness: {np.max(scores)}")

    for generation in range(100):

        # get new pop
        new_population = elitist_select(population, scores, 2)
        while len(new_population) < pop_size:
            candidates = tournament_select(rng, population, scores, 2)
            crossed_candidates = crossover(
                rng, candidates[0], candidates[1], crossover_prob
            )
            mutated_candidates = [
                mutate(rng, c, mutation_prob) for c in crossed_candidates
            ]
            new_population += mutated_candidates
        population = new_population

        # get new scores
        scores = [fitness_function(p, weights, values, max_weight) for p in population]

        print(f"Generation: {generation}")
        print(f"Average Fitness: {np.mean(scores)}")
        print(f"Max Fitness: {np.max(scores)}")
        print()

    # print(tournament_select(rng, population, scores, 4))
