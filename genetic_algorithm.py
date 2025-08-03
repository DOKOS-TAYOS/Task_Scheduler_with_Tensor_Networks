import numpy as np
from auxiliar_functions import checker, checker_sum
from optimizer import Optimizer, IterativeOptimizer

#-------------------------------------------------------------------------------
def correct_times(
    tasks_list: list[np.ndarray],
    costs_list: list[np.ndarray]
) -> list[np.ndarray]:
    """Correct processing times by extracting costs for active tasks on each machine.
    
    This function takes a list of active tasks for each machine and returns their corresponding
    processing costs. The tasks_list contains only the indices of active tasks, and the function
    uses these indices to look up the appropriate costs from costs_list.
    
    Args:
        tasks_list: List of numpy arrays containing indices of active tasks for each machine.
                   Each array contains only the task indices that are currently active.
        costs_list: List of numpy arrays containing the processing costs for all possible tasks
                   on each machine. Each array contains costs for all tasks, active or not.
        
    Returns:
        List of numpy arrays containing the processing costs for only the active tasks on each
        machine, in the same order as the input tasks_list
    """
    # Create copy of times and zero out inactive tasks
    new_costs = [np.array([t_machine[task] for task in tasks_list[i]]) 
                for i, t_machine in enumerate(costs_list)]

    return new_costs

#-------------------------------------------------------------------------------
def add_constraints(
    tasks_list: list,
    constraints: list,
    max_num_constraints: int
) -> list:
    """Add compatible constraint rules to an individual's ruleset.
    
    This function takes a list of task assignments and adds compatible constraint rules up to 
    the specified maximum number. It maps the original task indices to new indices based on
    the active tasks in tasks_list.
    
    Args:
        tasks_list: List of task assignments per machine
        constraints: List containing [conditions, targets] of possible constraint rules
        max_num_constraints: Maximum number of constraint rules to add
        
    Returns:
        List containing [conditions, targets] with the added compatible rules
    """
    # Create mapping from original task indices to new indices for each machine
    inverse_task_dict = [
        {int(v): int(k) for k, v in enumerate(task_machine)} 
        for task_machine in tasks_list
    ]

    # Initialize output lists and available rules
    conditions = []
    targets = []
    available_rules = set(range(len(constraints[0])))
    
    # Add compatible rules until max constraints or no more available
    while len(conditions) < max_num_constraints and available_rules:
        # Select random rule from available pool
        rule_idx = np.random.choice(list(available_rules))
        available_rules.remove(rule_idx)
        
        # Check if rule tasks are compatible with current task assignments
        rule_compatible = all(
            task is None or int(task) in tasks_list[machine]
            for machine, task in enumerate(constraints[0][rule_idx])
        ) and constraints[1][rule_idx][1] in tasks_list[constraints[1][rule_idx][0]]
        
        if rule_compatible:
            # Map rule tasks to new indices based on active tasks
            new_constraint = [
                None if task is None else inverse_task_dict[machine][task]
                for machine, task in enumerate(constraints[0][rule_idx])
            ]
            
            # Map target tasks to new indices
            new_target = constraints[1][rule_idx].copy()
            new_target[1] = inverse_task_dict[new_target[0]][new_target[1]]
            
            # Add mapped rules to output
            conditions.append(new_constraint)
            targets.append(new_target)

    return [conditions, targets]


def create_population(
    constraints: list[list[int|None]],
    costs_lists: list[np.ndarray], 
    num_indiv: int,
    max_num_constraints: int,
    num_used_tasks: int
) -> tuple[list, list, list]:
    """
    Creates an initial population for the genetic algorithm by randomly selecting task sets and rule sets.
    
    The function works by:
    1. First selecting a random subset of tasks to activate for each machine
    2. Then assigning compatible rules to each individual based on their active tasks
    3. Setting processing times of unused tasks to 0 to maintain consistency
    
    This approach has several advantages:
    - Selecting tasks before rules ensures rules only depend on active tasks
    - Each individual has its own subsystem of rules, partially decomposing the problem
    - Rules can be repeated across individuals
    - Setting unused task times to 0 allows for wider time windows
    
    Args:
        constraints: List of scheduling constraints [conditions, targets]
        costs_lists: List of processing costs for each task on each machine
        num_indiv: Number of individuals in population
        max_num_constraints: Maximum number of constraints per individual
        num_used_tasks: Number of tasks to activate per machine
        iter_max: Maximum iterations for rule assignment
        
    Returns:
        tuple[list, list, list]: Contains:
            - List of task activation patterns for each individual
            - List of processing costs for each individual
            - List of constraint sets assigned to each individual
    """
    # Get dimensions of the problem
    num_tasks = [len(t) for t in costs_lists]
    num_machines = len(costs_lists)

    # Initialize output lists
    tasks_indiv = []
    costs_indiv = []
    constraints_indiv = []

    # Generate population
    for _ in range(num_indiv):
        # Initialize task assignments by randomly selecting tasks for each machine
        task_values = [
            np.array(sorted(np.random.choice(num_tasks[machine], size=num_used_tasks, replace=False)), dtype=int) 
            for machine in range(num_machines)
        ]
        
        # Store task assignments and costs
        tasks_indiv.append(task_values)
        costs_indiv.append(correct_times(task_values, costs_lists))
        constraints_indiv.append(add_constraints(task_values, constraints, max_num_constraints))

    return tasks_indiv, costs_indiv, constraints_indiv

#-------------------------------------------------------------------------------

def process_offspring(
    num_crosses: int,
    num_machines: int, 
    parent1_idx: int,
    parent2_idx: int,
    offspring_indices: list[int],
    tasks_indiv: list[list[np.ndarray]],
    new_inds: list[list[np.ndarray]],
    new_costs: list[list[np.ndarray]],
    costs_list: list[np.ndarray[float]],
    new_constraints: list[list[list]],
    constraints: list[list],
    max_num_constraints: int
) -> None:
    """Helper function to process a pair of parents and create offspring.
    
    Performs crossover between two parent individuals to create offspring by:
    1. Copying parent task assignments to create initial offspring
    2. Swapping selected tasks between offspring
    3. Maintaining sorted task order and updating costs/constraints
    
    Args:
        num_crosses: Number of crossover points
        num_machines: Number of machines in the problem
        parent1_idx: Index of first parent
        parent2_idx: Index of second parent 
        offspring_indices: Indices for new offspring
        tasks_indiv: List of parent task assignments
        new_inds: List to store new offspring
        new_costs: List to store offspring costs
        costs_list: Processing costs for tasks
        new_constraints: List to store offspring constraints
        constraints: Problem constraints
        max_num_constraints: Max constraints per individual
    """
    # Get active tasks for each parent
    tasks1 = [machine_tasks.copy() for machine_tasks in tasks_indiv[parent1_idx]]
    tasks2 = [machine_tasks.copy() for machine_tasks in tasks_indiv[parent2_idx]]

    # Validate task lists
    for machine_tasks, parent_name in [(tasks1, 'tasks1'), (tasks2, 'tasks2')]:
        if any(len(tasks) == 0 for tasks in machine_tasks):
            raise ValueError(f"Empty task list found in {parent_name}: {machine_tasks}")

    # Create offspring by copying parents
    for idx in offspring_indices:
        if idx >= len(new_inds):
            continue
        new_inds[idx] = [np.copy(x_ind) for x_ind in tasks_indiv[parent1_idx if idx == offspring_indices[0] else parent2_idx]]

    # Perform crossovers
    for _ in range(num_crosses):
        # Select random machine and tasks to swap
        mach = np.random.choice(num_machines)
        task1 = np.random.choice(tasks1[mach])
        task2 = np.random.choice(tasks2[mach])
        
        can_swap = False

        if task2 not in new_inds[offspring_indices[0]][mach]:
            if len(offspring_indices) == 2:
                if task1 not in new_inds[offspring_indices[1]][mach]:
                    can_swap = True
            else:
                can_swap = True

        if can_swap:
            # Swap tasks between offspring
            for idx, (t1, t2) in zip(offspring_indices, [(task1, task2), (task2, task1)]):
                if idx < len(new_inds):
                    task_idx = np.where(new_inds[idx][mach] == t1)[0][0]
                    new_inds[idx][mach][task_idx] = t2

            # Update parent task lists
            for tasks, old_task, new_task in [(tasks1, task1, task2), (tasks2, task2, task1)]:
                tasks[mach] = np.delete(tasks[mach], np.where(tasks[mach] == old_task))
                tasks[mach] = np.append(tasks[mach], new_task)



    # Process each valid offspring
    for idx in offspring_indices:
        if idx >= len(new_inds):
            continue
            
        # Sort tasks for each machine
        for mach in range(num_machines):
            new_inds[idx][mach] = np.sort(new_inds[idx][mach])
        
        # Update costs and constraints
        new_costs[idx] = correct_times(new_inds[idx], costs_list)
        new_constraints[idx] = add_constraints(
            new_inds[idx],
            constraints,
            max_num_constraints
        )


def cross_indivs(
    tasks_indiv: list[list[np.ndarray]],
    constraints: list[list],
    costs_list: list[np.ndarray], 
    max_num_constraints: int,
    num_indivs: int,
    num_crosses: int
) -> tuple[list, list, list]:
    """Perform crossover between pairs of individuals to create new offspring.
    
    This function performs crossover operations between pairs of parent individuals to generate new offspring.
    The crossover process involves:
    
    1. Exchanging tasks between parents while maintaining task-rule compatibility
    2. Transferring compatible rules associated with exchanged tasks to offspring
    3. Mutating offspring by adding new compatible rules if needed
    4. Correcting processing times for inactive tasks
    
    The crossover maintains several key properties:
    - Tasks are exchanged only between machines of the same type
    - Rule compatibility is preserved for new task combinations
    - Processing times are adjusted for active/inactive tasks
    - The target number of rules per individual is maintained
    
    Args:
        tasks_indiv: List of parent individuals' task assignments
        constraints: List of constraints to enforce
        costs_list: List of processing costs for tasks
        max_num_constraints: Maximum number of constraints per individual
        num_indivs: Number of individuals in population
        num_crosses: Number of crossover points per pair
        
    Returns:
        tuple: Contains:
        - List of new offspring individuals
        - List of rules for new offspring
        - List of corrected processing times for offspring
    """
    # Initialize arrays for new generation
    num_tasks = [len(t) for t in costs_list]
    num_machines = len(num_tasks)
    new_inds = [[] for _ in range(num_indivs)]
    new_constraints = [[[],[]] for _ in range(num_indivs)]
    new_costs = [[] for _ in range(num_indivs)]

    

    # Process pairs of parents
    available_parents = list(range(num_indivs))
    for i in range(num_indivs // 2):
        # Select random pair of parents without replacement
        ind1, ind2 = np.random.choice(available_parents, size=2, replace=False)
        available_parents.remove(ind1)
        available_parents.remove(ind2)
        
        # Process offspring for first parent pair
        offspring_indices = [2*i, 2*i+1]
        process_offspring(
            num_crosses=num_crosses,
            num_machines=num_machines, 
            parent1_idx=ind1,
            parent2_idx=ind2,
            offspring_indices=offspring_indices,
            tasks_indiv=tasks_indiv,
            new_inds=new_inds,
            new_costs=new_costs,
            costs_list=costs_list,
            new_constraints=new_constraints,
            constraints=constraints,
            max_num_constraints=max_num_constraints
        )

    # Handle odd number of individuals
    if num_indivs % 2 != 0 and available_parents:
        ind1 = available_parents[0]
        ind2 = np.random.choice([i for i in range(num_indivs) if i != ind1])
        process_offspring(
            num_crosses=num_crosses,
            num_machines=num_machines,
            parent1_idx=ind1,
            parent2_idx=ind2,
            offspring_indices=[num_indivs-1],
            tasks_indiv=tasks_indiv,
            new_inds=new_inds,
            new_costs=new_costs,
            costs_list=costs_list,
            new_constraints=new_constraints,
            constraints=constraints,
            max_num_constraints=max_num_constraints
        )


    return new_inds, new_constraints, new_costs


def mutator_indivs(
    tasks_list: list[np.ndarray],
    constraints: list[list],
    costs_list: list[np.ndarray],
    num_rules: int,
    num_muts: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[list]]:
    """
    Mutate an individual by randomly swapping active and inactive tasks and updating rules.
    
    This function performs mutation in two steps:
    1. Task mutation:
       - Randomly selects machines and swaps active/inactive tasks
       - Updates processing times for the new task configuration
       - Checks rule compatibility with new task assignments
       
    2. Rule mutation:
       - Validates existing rules against new task configuration
       - Adds new compatible rules up to num_rules limit
       - May mutate first rule in ruleset if possible
    
    Args:
        tasks_list: List of numpy arrays containing active task indices for each machine
        constraints: List of scheduling constraints to enforce
        costs_list: List of numpy arrays containing processing costs for each task
        num_rules: Target number of rules to maintain
        num_muts: Number of task mutations to perform
        
    Returns:
        tuple containing:
            - list[np.ndarray]: New task assignments after mutation
            - list[np.ndarray]: Updated processing costs
            - list[list]: Updated compatible rules
    """
    num_tasks = [len(t) for t in costs_list]
    num_machines = len(num_tasks)
    
    # Create deep copies to avoid modifying originals
    tasks_copy = [np.copy(x_ind) for x_ind in tasks_list]
    new_individual_tasks = [np.copy(x_ind) for x_ind in tasks_list]

    # Perform mutations
    for _ in range(num_muts):
        # Select random machine
        mach = np.random.choice(num_machines)
        
        # Select random active task to deactivate
        task1 = np.random.choice(len(new_individual_tasks[mach]))
        
        # Select random inactive task to activate
        inactive_tasks = [t for t in range(num_tasks[mach]) if t not in tasks_copy[mach]]
        if inactive_tasks:
            task2 = np.random.choice(inactive_tasks)
            
            # Swap tasks
            new_individual_tasks[mach][task1] = task2
            
            # Update active tasks
            tasks_copy[mach] = np.delete(tasks_copy[mach], np.where(tasks_copy[mach] == task1))
            tasks_copy[mach] = np.append(tasks_copy[mach], task2)

    
    # Sort tasks for each machine
    for mach in range(num_machines):
        new_individual_tasks[mach] = np.sort(new_individual_tasks[mach])

    # Update costs and constraints
    new_costs = correct_times(new_individual_tasks, costs_list)
    new_constraints = add_constraints(
        new_individual_tasks,
        constraints,
        num_rules
    )

    return new_individual_tasks, new_costs, new_constraints

#----------------------------------------------------------------------------------------------
def generation_check_resolution(
    tasks_indiv: list[int],
    constraints: list[list[int|None]], 
    constraints_indivs_lists: list[list[int|None]],
    costs_lists: list[np.ndarray],
    costs_indivs_list: list[np.ndarray],
    tau: float,
    num_indiv: int,
    iterative: bool,
    parent_proportion: int
) -> tuple[list, list, list, list]:
    """Performs calculations and organization for each generation (Survival phase).
    
    Evaluates each individual in the population by:
    1. Running tensor network optimization to find task assignments
    2. Translating results back to original task indices
    3. Checking constraint satisfaction and calculating costs
    4. Selecting the best valid solutions as parents
    
    Args:
        tasks_indiv: List of task activation patterns for each individual
        constraints: Global scheduling constraints to enforce
        constraints_indivs_lists: List of constraint subsets for each individual
        costs_lists: Global processing costs for each task on each machine
        costs_indivs_list: List of processing costs for each individual
        tau: Temperature parameter for tensor network optimization
        num_indiv: Number of individuals in population
        iterative: Whether to use iterative optimization (default: False)
        parent_proportion: Factor to determine number of parents to select (default: 3)
        
    Returns:
        tuple[list, list, list, list]: Contains:
            - List of selected parent task patterns
            - List of constraint subsets for selected parents  
            - List of processing costs for selected parents
            - List of optimized task assignments for selected parents
            
    Note:
        The function performs the following key steps:
        1. Optimizes each individual's task assignments using tensor networks
        2. Maps optimized solutions back to original task indices
        3. Validates solutions against global constraints
        4. Calculates total processing costs
        5. Selects best valid solutions as parents for next generation
    """
    num_machines = len(costs_lists)
    target_size = num_indiv // parent_proportion
    
    # Calculate results and translate to original task indices
    results = []
    translated_results = []
    costs = np.zeros(num_indiv)
    check_list = []

    # Process each individual
    for i in range(num_indiv):
        # Get solution from optimizer
        if iterative:
            result = IterativeOptimizer(constraints_indivs_lists[i], costs_indivs_list[i], tau).minimize()
        else:
            result = Optimizer(constraints_indivs_lists[i], costs_indivs_list[i], tau).minimize()

        if -1 in result:
            for machine in range(len(results)):
                if results[machine] == -1:
                    results[machine] = 0
        results.append(result)
        
        # Map back to original task indices
        translated = np.zeros(num_machines, dtype=int)
        for machine in range(num_machines):
            task_idx = result[machine]
            translated[machine] = tasks_indiv[i][machine][task_idx]
        translated_results.append(translated)
        
        # Check constraints and calculate cost
        check_list.append(checker_sum(constraints, translated))
        costs[i] = sum(costs_lists[m][translated[m]] for m in range(num_machines))

    # Split into valid and invalid solutions
    valid_solution = [check[0] for check in check_list]
    incomp_solution = [check[1] for check in check_list]
    
    # Store valid solutions
    valid_solutions = []
    for i, is_valid in enumerate(valid_solution):
        if is_valid:
            valid_solutions.append({
                'ind': [np.copy(elem) for elem in tasks_indiv[i]],
                'rules': [rule.copy() for rule in constraints_indivs_lists[i]],
                'times': [t.copy() for t in costs_indivs_list[i]], 
                'cost': costs[i],
                'result': np.copy(translated_results[i])
            })
    
    v_len = len(valid_solutions)
    if v_len >= target_size:
        # Sort valid solutions by cost and take best ones
        valid_solutions.sort(key=lambda x: x['cost'])
        selected = valid_solutions[:target_size]
        return (
            [s['ind'] for s in selected],
            [s['rules'] for s in selected],
            [s['times'] for s in selected],
            [s['result'] for s in selected]
        )
    
    else:
        # Get indices of invalid solutions sorted by incompatibility then cost
        invalid_indices = [i for i, valid in enumerate(valid_solution) if not valid]
        invalid_sorted = sorted(
            invalid_indices,
            key=lambda i: (incomp_solution[i], costs[i])
        )[:target_size - v_len]
        
        # Combine valid solutions with best invalid ones
        parents = [s['ind'] for s in valid_solutions]
        parent_rules = [s['rules'] for s in valid_solutions]
        parent_times = [s['times'] for s in valid_solutions]
        parent_results = [s['result'] for s in valid_solutions]
        
        for i in invalid_sorted:
            parents.append([np.copy(elem) for elem in tasks_indiv[i]])
            parent_rules.append([rule.copy() for rule in constraints_indivs_lists[i]])
            parent_times.append([t.copy() for t in costs_indivs_list[i]])
            parent_results.append(np.copy(translated_results[i]))

        return parents, parent_rules, parent_times, parent_results


def genetic_optimizer(
    constraints: list[list[int|None]],
    costs_list: list[np.ndarray],
    tau: float,
    num_indivs: int,
    num_generations: int,
    max_num_constraints: int,
    num_used_tasks: int,
    num_crosses: int,
    num_muts: int,
    iterative: bool=False,
    parent_proportion: int = 3,
    verbose: bool = True
) -> list[np.ndarray]:
    """Run genetic algorithm for specified number of generations.
    
    This function implements a generational genetic algorithm that evolves a population of solutions
    through selection, crossover and mutation operations.
    
    The population in each generation consists of:
    - 1/parent_proportion surviving parents from previous generation 
    - 1/parent_proportion offspring created through crossover
    - 1/parent_proportion mutated individuals
    - Any remaining slots filled with new random individuals
    
    Args:
        constraints: List of scheduling constraints to enforce
        costs_list: List of processing costs for each task on each machine
        tau: Time window parameter for scheduling
        num_indivs: Total population size
        num_generations: Number of generations to run
        max_num_constraints: Maximum number of constraints per individual
        num_used_tasks: Number of active tasks per machine
        num_crosses: Number of crossover points per parent pair
        num_muts: Number of mutations per individual
        iterative: Whether to use iterative optimization (default False)
        parent_proportion: Parent population proportion (default 3)
        verbose: Print progress messages (default True)
        
    Returns:
        List of best solutions from final generation, ordered by fitness
    """
    # Calculate number of parents to keep each generation
    num_parents = num_indivs // parent_proportion
    
    # Initialize population
    tasks_indiv, costs_indivs_list, constraints_indivs_list = create_population(
        constraints, costs_list, num_indivs, max_num_constraints, num_used_tasks
    )

    if verbose:
        print('Population initialized.                    ', end='', flush=True)

    # Initial generation evaluation
    parents_tasks, parent_rules, parent_times, parent_results = generation_check_resolution(
        tasks_indiv, constraints, constraints_indivs_list, costs_list, costs_indivs_list, 
        tau, num_indivs, iterative, parent_proportion
    )
    # Main generational loop
    for generation in range(num_generations):
        # Create offspring through crossover
        children_tasks, children_constraints, children_costs = cross_indivs(
            parents_tasks, constraints, costs_list, 
            max_num_constraints, num_parents, num_crosses
        )

        # Create mutated individuals
        mutated_tasks, mutated_costs, mutated_constraints = [], [], []
        for parent in parents_tasks:
            mut_tasks, mut_costs, mut_rules = mutator_indivs(
                parent, constraints, costs_list, max_num_constraints, num_muts
            )
            mutated_tasks.append(mut_tasks)
            mutated_costs.append(mut_costs)
            mutated_constraints.append(mut_rules)

        # Combine parent, offspring and mutated populations
        tasks_indiv = parents_tasks + children_tasks + mutated_tasks
        constraints_indivs_list = parent_rules + children_constraints + mutated_constraints
        costs_indivs_list = parent_times + children_costs + mutated_costs

        # Fill remaining slots with new random individuals if needed
        remaining_slots = num_indivs - len(tasks_indiv)
        if remaining_slots > 0:
            new_tasks, new_costs, new_rules = create_population(
                constraints, costs_list, remaining_slots,
                max_num_constraints, num_used_tasks
            )
            tasks_indiv.extend(new_tasks)
            constraints_indivs_list.extend(new_rules)
            costs_indivs_list.extend(new_costs)

        # Evaluate generation and select parents for next generation
        parents_tasks, parent_rules, parent_times, parent_results = generation_check_resolution(
            tasks_indiv, constraints, constraints_indivs_list, costs_list, costs_indivs_list,
            tau, num_indivs, iterative, parent_proportion
        )

        if verbose:
            print(f'\rGeneration {generation + 1}/{num_generations}                          ', end='', flush=True)

    if verbose:
        print('\rOptimization completed.                                        ')

    return parent_results

def basic_use_genetic(
    constraints: list[list[int|None]],
    costs_list: list[np.ndarray],
    tau: float,
    num_indivs: int,
    num_generations: int,
    max_num_constraints: int,
    num_used_tasks: int,
    num_crosses: int,
    num_muts: int,
    iterative: bool=False,
    parent_proportion: int = 3,
    verbose: bool = True) -> tuple[np.ndarray, float]:
    """Run basic tensor network optimization for task scheduling.
    
    Args:
        constraints: List of scheduling constraints to enforce
        costs_list: List of arrays containing processing costs for each task on each machine
        tau: Time scaling factor for imaginary time evolution
        
    Returns:
        tuple:
            - np.ndarray: Optimized task schedule
            - float: Total processing cost (np.inf if solution invalid)
    """
    # Run tensor network optimization
    result = genetic_optimizer(
        constraints,
        costs_list,
        tau,
        num_indivs,
        num_generations,
        max_num_constraints,
        num_used_tasks,
        num_crosses,
        num_muts,
        iterative=iterative,
        parent_proportion=parent_proportion,
        verbose=verbose
    )[0]

    print('Tensor Network Solution:', result)

    # Validate solution and calculate total processing time
    is_valid, violations = checker(constraints, result)

    if is_valid:
        total_cost = sum(costs_list[i][int(result[i])] for i in range(len(costs_list)))
        print('Total Processing Time:', total_cost)
        return result, total_cost
    else:
        print('Solution violates scheduling constraints')
        print('Violations:', violations)
        return result, np.inf