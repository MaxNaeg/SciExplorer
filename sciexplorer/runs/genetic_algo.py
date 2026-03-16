

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable



def run_genetic_algorithm(initial_population:list[tuple[Any, float]],
                          selection_function:Callable,
                          mutate_function:Callable,
                          score_function:Callable,
                          steps:int = 100,
                          mutate_n:int = 2,
                          n_parallel:int = 1,
                          desired_score:float = None,
                          max_population_size:int = None,
                          removal_style: str = 'worst',
                          mutation_timeout: int = None,
                          ) -> Any:
    """Run a generic parallelizable steady-state genetic algorithm.
    Args:
        initial_population (list[tuple[Any, float]]): 
            List containing tuples of individuals and their scores.
        selection_function (function): 
            Function with signature (population: list[tuple[Any, float]], mutate_n: int) -> list[tuple[Any, float]], to select mutate_n individuals for reproduction.
        mutate_function (function): 
            Function with signature (individuals: list[tuple[Any, float]]) -> list[Any], 
            to mutate the individuals returned by the selection function.
            Might return only one individual.
        score_function (function): 
            Function with signature (individual: Any) -> tuple[Any, float] return a scored individual and score.
        generations (int, optional): Number of generations to run. Defaults to 100.
        mutate_n (int, optional): Number of individuals to mutate together. Defaults to 2.
        n_parallel (int, optional): Number of parallel processes to use. Defaults to 1.
        desired_score (float, optional): Desired score to reach. If reached, the algorithm stops early. Defaults to None.
        max_population_size (int, optional): Maximum population size. If set, the population is trimmed to this size. Defaults to None.
        removal_style (str, optional): Strategy to remove individuals when max_population_size is reached. Options: 'worst' (remove worst individuals), 'oldest' (remove oldest individuals).
        mutation_timeout (int, optional): Timeout in seconds for mutation and scoring of an individual. If exceeded, the individual is skipped. Defaults to None.
    Returns:
        Any: Best individual found.
        list[Any]: Final population.
        list[float]: Best score per generation.
        list[float]: All scores.
    """

    assert removal_style in ['worst', 'oldest'], "removal_style must be either 'worst' or 'oldest'"
    
    best_score_history = []
    all_scores = []
    if initial_population:
        population = initial_population
    else:
        population = []

    # Helper function to mutate and score individuals parallelly
    def mutate_and_score(individuals):
        mutated = mutate_function(individuals)
        scored, score = score_function(mutated)
        return (scored, score)
    
    total_submitted = 0
    futures = set()

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        # Submit initial batch of futures
        while total_submitted < min(n_parallel, steps):
            parents = selection_function(population, mutate_n)
            f = executor.submit(mutate_and_score, parents)
            futures.add(f)
            total_submitted += 1

    
        # Process futures as they complete
        while futures:
            for f in as_completed(futures):
                try:
                    if mutation_timeout:
                        scored_offspring = f.result(timeout=mutation_timeout)
                    else:
                        scored_offspring = f.result()
                except Exception as e:
                    # Task exceeded timeout; skip it
                    print(f"Exception during mutation and scoring: {e.__class__.__name__}: {e}\nSkipping this individual.")
                    futures.remove(f)
                    continue

                population.append(scored_offspring)
                all_scores.append(scored_offspring[1])

                # Track best score so far
                best_score = max(population, key=lambda x: x[1])[1]
                best_score_history.append(best_score)

                # Early stopping if desired score is reached
                if desired_score is not None and best_score >= desired_score:
                    print(f"<<Desired score {desired_score} reached with score {best_score}. Stopping early.>>")
                    return max(population, key=lambda x: x[1])[0], population, best_score_history, all_scores
                
                # Trim population if it exceeds max sizes
                if max_population_size is not None and len(population) > max_population_size:
                    if removal_style == 'oldest':
                        population = population[-max_population_size:]
                    elif removal_style == 'worst':
                        population = sorted(population, key=lambda x: x[1], reverse=True)[:max_population_size]
                    else:
                        raise ValueError(f"Unknown removal_style: {removal_style}")
                    

                futures.remove(f)
                # Submit a new job if steps remain
                if total_submitted < steps:
                    parents = selection_function(population, mutate_n)
                    new_f = executor.submit(mutate_and_score, parents)
                    futures.add(new_f)
                    total_submitted += 1

    best_individual = max(population, key=lambda x: x[1])[0]
    return best_individual, population, best_score_history, all_scores
