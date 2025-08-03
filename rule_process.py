import numpy as np
from typing import Any, Union
import tensornetwork as tn

def start_end(constraint: list[int|None]) -> tuple[int, int, Union[int, None], Union[int, None]]:############################
    """Find the first and last non-empty elements in a constraint.
    
    Scans a constraint list to find the indices and values of the first and last non-None elements.
    This is used to determine the range of machines involved in a scheduling rule.
    
    Args:
        constraint: List of task indices where None represents no task constraint
        
    Returns:
        tuple:
            - int: Index of first non-None machine (None if all None)
            - int: Index of last non-None machine (None if all None)
            - int|None: Task value at first non-None position (None if all None) 
            - int|None: Task value at last non-None position (None if all None)
            
    Example:
        >>> start_end([None, 1, None, 2, None])
        (1, 3, 1, 2)
    """
    first_machine = next((i for i, elem in enumerate(constraint) if elem is not None), None)
    final_machine = len(constraint) - 1 - next((i for i, elem in enumerate(reversed(constraint)) if elem is not None), None)
    
    first_task = constraint[first_machine] if first_machine is not None else None
    final_task = constraint[final_machine] if final_machine is not None else None
    
    return first_machine, final_machine, first_task, final_task


def sort_rules(constraint: list[list[int|None]]) -> tuple[list[list[int|None]], dict]:##################################################3
    """Sort and group rules based on machine usage and rule characteristics.

    This function organizes rules by machine usage patterns and rule properties to optimize processing.
    It performs an expensive one-time sorting operation to group related rules together.

    The input format is:
    constraint = [conditions, targets] where:
        - conditions[rule_number] contains the task conditions for each machine
        - targets[rule_number] contains [target_machine, target_task] pairs

    The output format is:
    sorted_rules[first_machine][rule_number] = [condition, target] where:
        - first_machine includes both initial machine and projector machine
        - condition contains the ordered task requirements
        - target contains the [machine, task] projection target

    The function also returns reordered machine indices to maintain alignment with processing times.

    Args:
        constraint: List containing [conditions, targets] rule specifications
        
    Returns:
        tuple:
            - list: Rules grouped and sorted by first involved machine
            - dict: Mapping between old and new machine indices
    """
    num_rules = len(constraint[0])
    num_machines = len(constraint[0][0])
    
    # Step 1: Sort machines by frequency of appearance in rules
    counter_machine = np.zeros(num_machines, dtype=int)
    for condition in constraint[0]:
        counter_machine += np.array([task is not None for task in condition], dtype=int)
                
    # Get sorted machine indices in descending order of frequency
    new_indexes = np.argsort(counter_machine)[::-1]
    new_old_machines = {int(k): int(v) for k, v in enumerate(new_indexes)}
    old_new_machines = {int(v): int(k) for k, v in new_old_machines.items()}
    
    # Rewrite rules with new machine ordering
    sorted_conditions = [
        [condition[new_old_machines[i]] for i in range(num_machines)]
        for condition in constraint[0]
    ]
    sorted_targets = [
        [old_new_machines[target[0]], target[1]]
        for target in constraint[1]
    ]
        
    # Step 2: Sort rules by number of conditions
    lengths = np.array([sum(task is not None for task in rule) for rule in sorted_conditions])
    sorted_indices = np.argsort(lengths)
    sorted_conditions = [sorted_conditions[i] for i in sorted_indices]
    sorted_targets = [sorted_targets[i] for i in sorted_indices]
    
    # Step 3: Sort by projector position
    projector_pos = np.array([target[0] for target in sorted_targets])
    sorted_indices = np.argsort(projector_pos)
    sorted_conditions = [sorted_conditions[i] for i in sorted_indices]
    sorted_targets = [sorted_targets[i] for i in sorted_indices]
    
    # Step 4: Sort by maximum distance between rule elements
    extreme_distances = []
    for condition, target in zip(sorted_conditions, sorted_targets):
        start_mach, final_mach, _, _ = start_end(condition)
        # Get maximum distance between target and rule endpoints
        max_distance = max(
            abs(target[0] - start_mach),
            abs(target[0] - final_mach), 
            abs(start_mach - final_mach)
        )
        extreme_distances.append(max_distance)
            
    sorted_indices = np.argsort(extreme_distances)
    sorted_conditions = [sorted_conditions[i] for i in sorted_indices]
    sorted_targets = [sorted_targets[i] for i in sorted_indices]
    
    # Step 5: Group rules by first involved machine
    grouped_constraints = [[] for _ in range(num_machines)]
    for condition, target in zip(sorted_conditions, sorted_targets):
        start_mach, _, _, _ = start_end(condition)
        # First machine is minimum of start machine and target machine
        first_machine = min(target[0], start_mach)
        grouped_constraints[first_machine].append([condition, target])
    
    return grouped_constraints, new_old_machines


def search_rules(constraints: list[list[Any]], num_tasks: list[int], basic_rule_scheme: list[Any]) -> tuple[list[Any], list[Any]]:
    """
    Search for compatible rules based on a given syntax.
    
    Args:
        constraints: List of constraint rules
        num_tasks: List of number of tasks per machine
        syntax: Rule syntax to match against [condition, target]
        
    Returns:
        tuple:
            - list: New compatible rules found
            - list: Rules that were erased/used
    """
    condition = basic_rule_scheme[0]
    target_machine = basic_rule_scheme[1][0]
    
    initial_machine, final_machine, initial_task, final_task = start_end(condition)
    new_rules = [basic_rule_scheme]

    # Handle single control case
    if initial_machine == final_machine:
        if target_machine < initial_machine:  # Projector on left
            useful_tasks = np.delete(np.arange(num_tasks[final_machine]), final_task)
            erased_rules = [target_machine, []]
            
            # Search rules with same final machine and projector
            for i, rule in enumerate(constraints[target_machine]):
                _, _final_machine, _, _final_task = start_end(rule[0])
                if (_final_machine == final_machine and 
                    rule[1][0] == target_machine and 
                    _final_task in useful_tasks):
                    new_rules.append(rule)
                    useful_tasks = np.setdiff1d(useful_tasks, [_final_task])
                    erased_rules.append(i)
                    if len(useful_tasks) == 0:
                        return new_rules, erased_rules
                        
        else:  # Projector on right
            useful_tasks = np.delete(np.arange(num_tasks[initial_machine]), initial_task)
            erased_rules = [initial_machine, []]
            
            # Search rules before projector
            for i, rule in enumerate(constraints[initial_machine]):
                _, _final_machine, _initial_task, _ = start_end(rule[0])
                if (_final_machine < target_machine and 
                    rule[1][0] == target_machine and 
                    _initial_task in useful_tasks):
                    new_rules.append(rule)
                    useful_tasks = np.setdiff1d(useful_tasks, [_initial_task])
                    erased_rules.append(i)
                    if len(useful_tasks) == 0:
                        return new_rules, erased_rules

    # Handle multiple controls case            
    else:
        if target_machine < initial_machine:  # Projector on left
            useful_tasks = np.delete(np.arange(num_tasks[final_machine]), final_task)
            erased_rules = [target_machine, []]
            
            for i, rule in enumerate(constraints[target_machine]):
                _, _final_machine, _, _final_task = start_end(rule[0])
                if (_final_machine == final_machine and 
                    rule[1][0] == target_machine and 
                    _final_task in useful_tasks):
                    new_rules.append(rule)
                    useful_tasks = np.setdiff1d(useful_tasks, [_final_task])
                    erased_rules.append(i)
                    if len(useful_tasks) == 0:
                        return new_rules, erased_rules

        elif final_machine < target_machine:  # Projector on right
            useful_tasks = np.delete(np.arange(num_tasks[initial_machine]), initial_task)
            erased_rules = [initial_machine, []]
            
            for i, rule in enumerate(constraints[initial_machine]):
                _, _final_machine, _initial_task, _ = start_end(rule[0])
                if (_final_machine < target_machine and 
                    rule[1][0] == target_machine and 
                    _initial_task in useful_tasks):
                    new_rules.append(rule)
                    useful_tasks = np.setdiff1d(useful_tasks, [_initial_task])
                    erased_rules.append(i)
                    if len(useful_tasks) == 0:
                        return new_rules, erased_rules

        else:  # Projector in middle
            match = condition[:target_machine]
            useful_tasks = np.delete(np.arange(num_tasks[final_machine]), final_task)
            erased_rules = [initial_machine, []]
            
            for i, rule in enumerate(constraints[initial_machine]):
                if rule[0][:target_machine] == match:
                    _, _final_machine, _, _final_task = start_end(rule[0])
                    if (_final_machine == final_machine and 
                        rule[1][0] == target_machine and 
                        _final_task in useful_tasks):
                        new_rules.append(rule)
                        useful_tasks = np.setdiff1d(useful_tasks, [_final_task])
                        erased_rules.append(i)
                        if len(useful_tasks) == 0:
                            return new_rules, erased_rules
    
    return new_rules, erased_rules


def tensor_denser(machine_instruction: list[list[str]], num_tasks: int, num_rules: int, target_position: str) -> np.ndarray:
    """Create dense tensor representation based on machine instructions.
    
    Args:
        machine_instruction: List of instructions for each machine
        num_tasks: Number of tasks per machine
        num_rules: Number of rules to process
        target_position: Position of target machine ('start', 'middle', or 'end')
        
    Returns:
        ndarray: Dense tensor representation
    """
    # Handle end target position
    if target_position == 'end':
        # Determine tensor shape and type based on first instruction
        instr_type = machine_instruction[0][0]
        if instr_type in ('Ctrld', 'Proyi'):
            megatensor = np.zeros((num_tasks, num_tasks, num_rules+1), dtype=complex)
            is_ctrl = instr_type == 'Ctrld'
            
            # Initialize identity matrix
            np.fill_diagonal(megatensor[:,:,0], 1)
            
            # Process instructions
            for i, instr in enumerate(machine_instruction):
                if is_ctrl:
                    megatensor[int(instr[1]), int(instr[1]), 0] = 0
                megatensor[int(instr[1]), int(instr[1]), i+1] = 1
                
        else: # Cctrld or Passd
            megatensor = np.zeros((num_tasks, num_tasks, num_rules+1, num_rules+1), dtype=complex)
            
            # Initialize identity matrix
            np.fill_diagonal(megatensor[:,:,0,0], 1)
            
            # Process instructions
            for i, instr in enumerate(machine_instruction):
                if instr[0] == 'Cctrld':
                    # Set up control
                    megatensor[:,:,i+1,0] = np.eye(num_tasks)
                    megatensor[int(instr[1]), int(instr[1]), i+1, 0] = 0
                    megatensor[int(instr[1]), int(instr[1]), i+1, i+1] = 1
                else: # Passd
                    np.fill_diagonal(megatensor[:,:,i+1,i+1], 1)
                    
    else: # Handle other target positions
        # Determine tensor shape and type
        instr_type = machine_instruction[0][0]
        
        if instr_type == 'Ctrld':
            megatensor = np.zeros((num_tasks, num_tasks, 2), dtype=complex)
            np.fill_diagonal(megatensor[:,:,0], 1)
            instr = machine_instruction[0]
            megatensor[int(instr[1]), int(instr[1]), 0] = 0
            megatensor[int(instr[1]), int(instr[1]), 1] = 1
            
        elif instr_type == 'Proyd':
            megatensor = np.zeros((num_tasks, num_tasks, num_rules+1), dtype=complex)
            np.fill_diagonal(megatensor[:,:,0], 1)
            for i, instr in enumerate(machine_instruction):
                megatensor[int(instr[1]), int(instr[1]), i+1] = 1
                
        elif instr_type == 'Ctrli':
            megatensor = np.zeros((num_tasks, num_tasks, num_rules+1), dtype=complex)
            np.fill_diagonal(megatensor[:,:,0], 1)
            for i, instr in enumerate(machine_instruction):
                megatensor[int(instr[1]), int(instr[1]), 0] = 0
                megatensor[int(instr[1]), int(instr[1]), i+1] = 1
                
        elif instr_type in ('Cctrld', 'Passd'):
            megatensor = np.zeros((num_tasks, num_tasks, 2, 2), dtype=complex)
            np.fill_diagonal(megatensor[:,:,0,0], 1)
            
            if instr_type == 'Cctrld':
                instr = machine_instruction[0]
                megatensor[:,:,1,0] = np.eye(num_tasks)
                megatensor[int(instr[1]), int(instr[1]), 1, 0] = 0
                megatensor[int(instr[1]), int(instr[1]), 1, 1] = 1
            else:
                np.fill_diagonal(megatensor[:,:,1,1], 1)
                
        elif instr_type in ('Cctrli', 'Passi'):
            megatensor = np.zeros((num_tasks, num_tasks, num_rules+1, num_rules+1), dtype=complex)
            np.fill_diagonal(megatensor[:,:,0,0], 1)
            
            for i, instr in enumerate(machine_instruction):
                if instr[0] == 'Cctrli':
                    megatensor[:,:,0,i+1] = np.eye(num_tasks)
                    megatensor[int(instr[1]), int(instr[1]), 0, i+1] = 0
                    megatensor[int(instr[1]), int(instr[1]), i+1, i+1] = 1
                else:
                    for j in range(num_rules):
                        np.fill_diagonal(megatensor[:,:,j+1,j+1], 1)
                        
        else: # cProy
            megatensor = np.zeros((num_tasks, num_tasks, 2, num_rules+1), dtype=complex)
            # Initialize identity and default states
            np.fill_diagonal(megatensor[:,:,0,0], 1)
            np.fill_diagonal(megatensor[:,:,1,0], 1)
            megatensor[:,:,0,1:] = np.eye(num_tasks)[:,:,np.newaxis]
            
            # Set projections
            for i, instr in enumerate(machine_instruction):
                megatensor[int(instr[1]), int(instr[1]), 1, i+1] = 1

    return megatensor


def join_rules(constraints: list[list[Any]], num_tasks: list[int], name: str = 'rule') -> tuple[list[tn.Node], int]:
    """Join multiple rules into a tensor network representation.

    This function unifies multiple rules into a tensor network by connecting compatible rules
    and applying directional control logic. It processes rules based on their machine dependencies
    and creates appropriate tensor dimensions.

    The function works by:
    1. Finding compatible rules that can be joined
    2. Creating tensors with appropriate dimensions based on rule type:
        - For 'd' type rules: Left index has dimension 1, right has dimension n_rules
        - For 'i' type rules: Both indices have dimension n_rules
        - For projectors: Dimension n_rules for 'd'/'i', but 1 and n_rules for 'cProy'
    3. Connecting the tensor nodes according to the control flow
    4. Applying projections at target machines

    All tensors maintain n dimensions for top and bottom indices to preserve task information.

    Args:
        constraints: List of constraint lists containing rules and targets
        num_tasks: List of number of tasks per machine
        name: Base name for tensor network nodes
        
    Returns:
        tuple:
            - list: Connected tensor network nodes representing the unified rules
            - int: Starting machine index for rule application
    """
    num_rules = len(constraints)
    num_machines = len(num_tasks)
    target_machine = constraints[0][1][0]
    
    # Find start and end machines involved
    start_point_of_layer, end, _, _ = start_end(constraints[0][0])
    start_point_of_layer = min(target_machine, start_point_of_layer)
    end = max(target_machine, end)
    
    # Determine target machine position
    if target_machine == start_point_of_layer:
        target_position = 'start'
    elif target_machine == end:
        target_position = 'end' 
    else:
        target_position = 'middle'
    
    # Initialize instruction array
    machine_instruction = list(np.zeros((num_rules, num_machines, 2), dtype='<U6'))
    
    # Fill instructions for each rule
    for i, rule in enumerate(constraints):
        # Process machines before target
        for j in range(start_point_of_layer, target_machine):
            if j == start_point_of_layer:
                machine_instruction[i][j] = ['Ctrld', int(rule[0][j])]
            else:
                if rule[0][j] == None:
                    machine_instruction[i][j] = ['Passd', None]
                else:
                    machine_instruction[i][j] = ['Cctrld', int(rule[0][j])]
                    
        # Process machines after target
        for j in range(target_machine + 1, end + 1):
            if j == end:
                machine_instruction[i][j] = ['Ctrli', int(rule[0][j])]
            else:
                if rule[0][j] == None:
                    machine_instruction[i][j] = ['Passi', None]
                else:
                    machine_instruction[i][j] = ['Cctrli', int(rule[0][j])]
        
        # Set target machine instruction
        target_instr = {
            'start': 'Proyd',
            'end': 'Proyi',
            'middle': 'cProy'
        }
        machine_instruction[i][target_machine] = [
            target_instr[target_position],
            int(rule[1][1])
        ]
            
    # Create tensors and nodes
    tensors = [
        tensor_denser(
            [instr[i] for instr in machine_instruction],
            num_tasks[i],
            num_rules,
            target_position
        )
        for i in range(start_point_of_layer, end + 1)
    ]
    
    # Create nodes with appropriate axes
    layer = []
    for i, tensor in enumerate(tensors):
        axis_names = [f'{name}_{i}_in', f'{name}_{i}_out']
        
        if i == 0:
            axis_names.append(f'{name}_{i}_right')
        elif i == end - start_point_of_layer:
            axis_names.append(f'{name}_{i}_left')
        else:
            axis_names.extend([f'{name}_{i}_left', f'{name}_{i}_right'])
            
        layer.append(tn.Node(tensor, name=f'{name}_{i}', axis_names=axis_names))
    
    # Connect nodes
    layer[0][2] ^ layer[1][2]
    for i in range(1, len(layer) - 1):
        layer[i][3] ^ layer[i + 1][2]
    
    return layer, start_point_of_layer

def erase_constraints(constraints_list: list[list[int|None]], idx_erased_const: list[int,list[int]]) -> list[list[int|None]]:###############################
    """Remove specified constraints from a list of constraints.
    
    Args:
        constraints_list: List of constraint lists to modify. Each inner list contains
                        constraints for a specific machine.
        idx_erased_const: List containing [machine_index, indices_to_erase] where
                         machine_index specifies which machine's constraints to modify and
                         indices_to_erase contains the indices of constraints to remove.
            
    Returns:
        list[list[int|None]]: Modified constraints list with specified constraints removed.
                             The original list is modified in-place.
    """
    # Extract machine index and indices to erase
    machine_idx = idx_erased_const[0]
    indices_to_erase = idx_erased_const[1]
    
    # Iterate through indices in reverse order to avoid shifting issues
    # when removing multiple elements from the list
    for idx in sorted(indices_to_erase, reverse=True):
        constraints_list[machine_idx].pop(idx)
        
    return constraints_list
