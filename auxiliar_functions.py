import numpy as np

def normalize_times(times_list: list[np.ndarray]) -> list[np.ndarray]:############################################################
    """
    Normalize production times to a distribution between -1 and 1.
    
    Args:
        times_list (list): List of lists containing production times for each machine
        
    Returns:
        tuple:
            - list: Normalized production times between 0 and 1 
    """
    # Get dimensions
    num_machines = len(times_list)
    
    # Initialize arrays
    t_max = np.array([max(times) for times in times_list], dtype=float)
    t_min = np.array([min(times) for times in times_list], dtype=float)
    
    # Calculate normalization factors
    total_max = t_max.sum()
    total_min = t_min.sum()
    scale = total_max - total_min

    offset = total_min / num_machines
    
    # Normalize times between 0 and 1
    # Vectorized normalization of all time arrays
    new_times_list = [(times - offset) / scale for times in times_list]
    
    return new_times_list

def create_rule(num_tasks: np.ndarray) -> tuple[list, list]:#####################################################################3
    """
    Creates a random rule for task scheduling.
    
    Generates a random scheduling rule by:
    1. Randomly assigning tasks to some machines (used machines)
    2. Leaving other machines empty (unused machines) 
    3. Selecting one unused machine as the target
    4. Assigning a random task to the target machine
    
    Args:
        num_tasks (np.ndarray): List containing number of tasks per machine
        
    Returns:
        tuple:
            - list: Rule conditions for each machine (None if not used)
            - list: Target machine and task [machine_idx, task_idx]
    """
    PROB_EMPTY = 0.5
    num_machines = len(num_tasks)
    
    while True:
        # Generate random rule assignments with negative values indicating unused machines
        # Scale probability of empty machines by 2*PROB_EMPTY
        condition = np.random.randint(
            low=-np.array(num_tasks, dtype=int) * (PROB_EMPTY * 2),
            high=np.array(num_tasks),
            size=num_machines
        )
        
        # Split into used and unused machines
        used, not_used, rule_final = [], [], []
        for i, val in enumerate(condition):
            if val < 0:
                rule_final.append(None)
                not_used.append(i)
            else:
                rule_final.append(int(val)) 
                used.append(i)
                
        # Only return if we have both used and unused machines
        if not_used and used:
            break
            
    # Pick random unused machine as target and assign random task
    target_mach = int(np.random.choice(not_used))
    target = [target_mach, int(np.random.randint(0, num_tasks[target_mach]))]
    
    return rule_final, target

#--------------------------------------------------------------------------------------------
def create_instance(
    num_constraints: int,
    num_machines: int, 
    tasks_per_mach: int,
    max_time: float = 10.0
) -> tuple[list[np.ndarray], list[list[int | None]]]:############################################################
    """
    Creates a random instance of a task scheduling problem.
    
    Args:
        num_constraints: Number of rules to generate
        num_machines: Number of machines
        tasks_per_mach: Number of tasks per machine
        max_time: Maximum task duration
        
    Returns:
        tuple:
            - list: Processing times for each task
            - list: Rules for task scheduling [conditions, targets]
            
    Raises:
        ValueError: If unable to generate unique rules with given parameters
    """
    # Constants
    MAX_ATTEMPTS_MULTIPLIER = 100
    
    # Initialize arrays
    num_tasks = np.array([tasks_per_mach] * num_machines)
    times_list = [np.random.uniform(0, max_time, size=tasks_per_mach) for _ in range(num_machines)]
    rule_conds = []
    rule_targets = []
    
    max_attempts = tasks_per_mach * MAX_ATTEMPTS_MULTIPLIER
    
    # Generate unique rules
    for i in range(num_constraints):
        attempts = 0
        
        # First rule doesn't need uniqueness check
        if i == 0:
            condition, target = create_rule(num_tasks)
            rule_conds.append(condition)
            rule_targets.append(target)
            continue
            
        # Try to generate unique rule
        while attempts < max_attempts:
            condition, target = create_rule(num_tasks)
            if condition not in rule_conds:
                rule_conds.append(condition)
                rule_targets.append(target)
                break
            attempts += 1
                
        if attempts >= max_attempts:
            raise ValueError(
                f'Failed to generate unique rule after {max_attempts} attempts. '
                'Try reducing num_constraints or increasing tasks_per_mach.'
            )
            
    return times_list, [rule_conds, rule_targets]

def checker_with_classified_rules(rules_list: list, solution: np.ndarray) -> tuple[bool, list[int]]:######################################
    """
    Check if a solution satisfies all rules in the machines classification, and return the rule that was violated if any.
    
    Args:
        rules_list: List of rules to check, where each rule contains [condition, target].
                   Condition specifies task assignments that trigger the rule.
                   Target specifies required task assignment when condition is met.
        solution: Array containing the current task assignments for each machine
        
    Returns:
        tuple:
            - bool: True if all rules satisfied, False if any rule violated 
            - list: [machine_index, rule_index] identifying the violated rule, or [-1, -1] if none violated
    """
    # Handle empty rules case
    if not rules_list:
        return True, [-1, -1]

    # Check each machine's rules
    for machine_idx, machine_rules in enumerate(rules_list):
        for rule_idx, rule in enumerate(machine_rules):
            condition, target = rule[0], rule[1]
            
            # Get list of (machine, task) pairs from condition, skipping None values
            condition_pairs = [
                (machine, int(task)) 
                for machine, task in enumerate(condition) 
                if task is not None
            ]
            
            # Check if all condition tasks are assigned correctly
            conditions_met = all(
                solution[machine] == task 
                for machine, task in condition_pairs
            )
            
            # If conditions met, verify target task assignment
            if conditions_met:
                target_machine = int(target[0])
                target_task = int(target[1])
                if solution[target_machine] != target_task:
                    return False, [machine_idx, rule_idx]
                
    return True, [-1, -1]

def checker(constraints_list: list, solution: np.ndarray) -> tuple[bool, int]:####################################################3
    """
    Check if a solution satisfies all rules and return which rule was violated if any.
    
    Args:
        rules_list: List containing [conditions, targets] for rules to check, where:
            - conditions: List of task assignments that trigger rules
            - targets: List of required task assignments when conditions are met
        solution: Array containing the current task assignments for each machine
        
    Returns:
        tuple:
            - bool: True if all rules satisfied, False if any rule violated
            - int: Index of violated rule, or -1 if none violated
    """
    if not constraints_list[0]:  # No rules case
        return True, -1

    conditions = constraints_list[0]
    targets = constraints_list[1]
    
    for rule_idx, condition in enumerate(conditions):
        # Get list of (machine, task) pairs from condition, skipping None values
        condition_pairs = [
            (machine, int(task)) 
            for machine, task in enumerate(condition) 
            if task is not None
        ]
        
        # Check if all condition tasks are assigned correctly
        conditions_met = all(
            solution[machine] == task 
            for machine, task in condition_pairs
        )
        
        # If conditions met, verify target task assignment
        if conditions_met:
            target_machine = int(targets[rule_idx][0])
            target_task = int(targets[rule_idx][1])
            if solution[target_machine] != target_task:
                return False, rule_idx
                
    return True, -1


def checker_sum(constraints_list: list, solution: np.ndarray) -> tuple[bool, int]:####################################################3
    """
    Check if a solution satisfies all rules and return which rule was violated if any.
    
    Args:
        rules_list: List containing [conditions, targets] for rules to check, where:
            - conditions: List of task assignments that trigger rules
            - targets: List of required task assignments when conditions are met
        solution: Array containing the current task assignments for each machine
        
    Returns:
        tuple:
            - bool: True if all rules satisfied, False if any rule violated
            - int: Index of violated rule, or -1 if none violated
    """
    if not constraints_list[0]:  # No rules case
        return True, -1

    conditions = constraints_list[0]
    targets = constraints_list[1]
    incompatibilities_count = 0
    
    for rule_idx, condition in enumerate(conditions):
        # Get list of (machine, task) pairs from condition, skipping None values
        condition_pairs = [
            (machine, int(task)) 
            for machine, task in enumerate(condition) 
            if task is not None
        ]
        
        # Check if all condition tasks are assigned correctly
        conditions_met = all(
            solution[machine] == task 
            for machine, task in condition_pairs
        )
        
        # If conditions met, verify target task assignment
        if conditions_met:
            target_machine = int(targets[rule_idx][0])
            target_task = int(targets[rule_idx][1])
            if solution[target_machine] != target_task:
                incompatibilities_count += 1
    if incompatibilities_count > 0:  
        return True, incompatibilities_count
    else:    
        return True, -1

#--------------------------------------------------------------------------------------------
def brute_force(rules_list: list, times_list: list[np.ndarray]) -> tuple[float, np.ndarray]:############################################
    """
    Performs exhaustive search to find a valid solution minimizing total time cost.
    
    Args:
        rules_list: List containing rules to check solutions against
        times_list: List of time costs for each task on each machine
        
    Returns:
        tuple:
            - float: Best cost found (total processing time)
            - ndarray: Best solution found (task assignments per machine)
    """
    # Get dimensions
    num_tasks_per_machine = [len(times) for times in times_list]
    num_machines = len(times_list)
    
    # Initialize best solution tracking
    best_cost = float('inf')
    best_solution = np.zeros(num_machines, dtype=int)
    
    # Try all possible task assignments
    for solution in np.ndindex(*num_tasks_per_machine):
        solution = np.array(solution)
        
        # Only consider solutions that satisfy all rules
        if checker(rules_list, solution)[0]:
            # Calculate total processing time
            total_time = sum(times[task] for times, task in zip(times_list, solution))
            
            # Update best solution if this one has lower cost
            if total_time < best_cost:
                best_cost = total_time
                best_solution = solution.copy()
                
    return best_cost, best_solution

def apply_gate(initial_tensor_network: list, index_list: np.ndarray, input_indexes: list, output_indexes: list) -> list:################################################3
    """
    Applies a quantum gate operation to specified indices of a tensor network.
    
    Args:
        initial_tensor_network: List of input tensor nodes to apply gate to
        index_list: Array of indices where gate should be applied
        input_indexes: List of gate input edges to connect
        output_indexes: List of gate output edges to connect
        
    Returns:
        list: Updated tensor network with gate applied at specified indices
    """
    # Connect gate inputs
    for gate_idx, tensor_idx in enumerate(index_list):
        initial_tensor_network[tensor_idx] ^ input_indexes[gate_idx]
        
    # Build output tensor network, replacing gated indices with outputs
    output_network = []
    for gate_idx in range(len(initial_tensor_network)):
        if gate_idx in index_list:
            # Replace gated node with corresponding output
            gate_position = np.where(index_list == gate_idx)[0][0]
            output_network.append(output_indexes[gate_position])
        else:
            # Keep ungated nodes unchanged
            output_network.append(initial_tensor_network[gate_idx])
            
    return output_network
