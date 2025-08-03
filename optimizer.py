import numpy as np
import tensornetwork as tn
from rule_process import sort_rules, search_rules, join_rules, erase_constraints, start_end
from auxiliar_functions import normalize_times, apply_gate, checker_with_classified_rules, checker, brute_force


def generate_plus_tensors(times_list: np.ndarray[float], tau: float, name: str) -> tn.Node:
    """Create a tensor network node representing initial state with time evolution.
    
    Creates a node with amplitudes determined by applying imaginary time evolution 
    to the processing times. The amplitudes are calculated as exp(-t*tau) where t
    are the processing times and tau is the evolution parameter.
    
    Args:
        times_list: Array of processing times for tasks
        tau: Time evolution parameter controlling amplitude decay
        name: Name identifier for the node
        
    Returns:
        tn.Node: Node containing the evolved state amplitudes with a single index
    """
    # Apply imaginary time evolution to get amplitudes
    tensor = np.exp(-np.array(times_list) * tau)
    
    # Create node with single index axis
    return tn.Node(tensor, name=name, axis_names=[f'{name}_index'])

def generate_id_node(dimension: int, name: str) -> tn.Node:
    """Create an identity node with specified dimension.
    
    Creates a tensor network node containing an identity matrix of the given dimension.
    The node has two axes - an input and output axis - which can be used to connect
    it to other nodes in the network.
    
    Args:
        dimension: Dimension of the identity matrix (number of rows/columns)
        name: Name identifier for the node, used to label axes
        
    Returns:
        tn.Node: Node containing identity matrix with input/output axes labeled as
                {name}_in and {name}_out
    """

    # Create node with input and output axes
    return tn.Node( np.eye(dimension, dtype=float), name=name, axis_names=[f'{name}_in', f'{name}_out'] )


class Optimizer:
    """Optimizer class for task scheduling using tensor networks.
    
    Handles optimization of task scheduling across multiple machines using tensor network
    techniques and rule-based constraints.
    """
    def __init__(self, constraints: list[list[int|None]], costs_list: list[np.ndarray], tau: float):
        """Initialize the optimizer.
        
        Args:
            constraints: List of scheduling rules/constraints to enforce. Each constraint is a list 
                       of task indices or None values indicating which tasks must be scheduled together.
                       An empty constraint list ([[]]) indicates no constraints.
            costs_list: List of arrays containing processing costs for each task on each machine.
                       Each array represents the costs for one machine. The arrays can have different
                       lengths since machines may handle different numbers of tasks.
            tau: Time scaling factor for imaginary time evolution. Controls how quickly the system
                 converges to optimal schedule. Larger values lead to faster convergence but may
                 miss the optimal solution.
            
        The optimizer uses tensor networks to find optimal task schedules that satisfy the given
        constraints while minimizing total processing costs. The tau parameter controls the 
        imaginary time evolution which helps find the ground state representing the optimal schedule.
        
        The tensor network is built up layer by layer, with each layer representing a set of
        compatible scheduling rules that can be applied simultaneously. Rules are considered
        compatible if they operate on non-overlapping sets of machines.
        
        The optimization process involves:
        1. Normalizing and reordering the input costs based on machine dependencies
        2. Building the tensor network layer by layer with compatible rule sets
        3. Contracting the network to find the optimal schedule that satisfies all constraints
        """
        # Basic initialization
        self.num_machines = len(costs_list)
        self.tau = tau

        # Handle case with no constraints
        if not constraints[0]:
            self.no_rules = True
            self.costs_list_sorted = costs_list
        else:
            self.no_rules = False
            costs_list_aux = normalize_times(costs_list)
            self.constraints_sorted, self.new_old_machines = sort_rules(constraints)
            self.old_new_machines = {int(v): int(k) for k,v in self.new_old_machines.items()}
            
            # Initialize per-machine data with reordered costs based on machine mapping
            self.costs_list_sorted = [costs_list_aux[self.new_old_machines[i]] for i in range(self.num_machines)]
            self.num_tasks_sorted = np.array([len(t) for t in self.costs_list_sorted], dtype=int)
            
            # Initialize tensor network structure
            self.layer_nodes = [self.num_machines]  # Nodes in each layer
            self.start_of_layer = [0]  # Starting node index for each layer
            self.total_tensors = [self.create_initial()]
            self.projection = self.create_trace()
            self.tensor_network_to_contract = []
            self.index = 0

    def create_initial(self) -> list[tn.Node]:
        """Create initial state tensors for each machine."""
        return [
            generate_plus_tensors(
                self.costs_list_sorted[i],
                self.tau,
                f'T_{i}'
            ) for i in range(self.num_machines)
        ]

    def create_trace(self) -> list[tn.Node]:
        """Create projection tensors for measurement."""
        return [
            generate_plus_tensors(
                self.costs_list_sorted[i],
                0,
                f'trace_{i}',
            ) for i in range(self.num_machines)
        ]

    def rules_cycle(self, basic_rule_scheme: list[int|None]) -> tuple[list[tn.Node], int]:
        """Process a single rules cycle by finding compatible rules, joining them into a layer,
        and updating the working constraints.
        
        Args:
            basic_rule_scheme: Rule syntax pattern to match against for finding compatible rules.
                             Contains conditions and target machine specifications.
            
        Returns:
            tuple:
                - list[tn.Node]: Tensor network nodes representing the joined rules layer
                - int: Starting machine index for this layer's rule application
        """
        # Find compatible rules that match the given scheme
        constraints_to_condense, idx_erased_const = search_rules(
            self.working_constraints, 
            self.num_tasks_sorted, 
            basic_rule_scheme
        )
        
        # Join compatible rules into a tensor network layer
        layer, start_point_of_layer = join_rules(
            constraints_to_condense,
            self.num_tasks_sorted, 
            name=f'Constraint_{self.index}'
        )
        
        # Remove used rules from working constraints
        erase_constraints(self.working_constraints, idx_erased_const)
        
        return layer, start_point_of_layer

    def connect_layers(self) -> None:
        """Join all rule layers into final tensor network.
        
        This method connects the tensor network layers by:
        1. Creating a flattened version of the network for auto-contraction
        2. Connecting adjacent layers using apply_gate operations
        3. Adding identity nodes in any remaining open edges
        4. Storing the final connected network for contraction
        """
        num_layers = len(self.total_tensors)
        
        # Create flattened network for auto-contraction
        tensor_network_flat = tn.replicate_nodes(self.total_tensors[0])
        
        # Calculate end indices for each layer
        end_of_layer = [
            self.start_of_layer[i] + len(layer) 
            for i, layer in enumerate(self.total_tensors)
        ]
        
        # Extract input nodes from first layer
        tensor_network_layers = [node[0] for node in tensor_network_flat]

        # Connect subsequent layers
        for i in range(1, num_layers):
            # Get indices for current layer
            tensor_idx = np.arange(self.start_of_layer[i], end_of_layer[i], dtype=int)
            
            # Replicate nodes and extract input/output edges
            nodes = tn.replicate_nodes(self.total_tensors[i])
            inputs_idx = [node[0] for node in nodes]
            outputs_idx = [node[1] for node in nodes]
            
            # Connect layer using apply_gate
            tensor_network_layers = apply_gate(
                tensor_network_layers, 
                tensor_idx, 
                inputs_idx, 
                outputs_idx
            )
            tensor_network_flat.extend(nodes)

        # Add identity nodes to close any open edges
        identities = [
            generate_id_node(self.num_tasks_sorted[i], f'Id_{i}') 
            for i in range(self.num_machines)
        ]
        tensor_idx = np.arange(self.num_machines, dtype=int)
        inputs_idx = [node[0] for node in identities]
        outputs_idx = [node[1] for node in identities]
        
        # Connect identity nodes
        tensor_network_layers = apply_gate(
            tensor_network_layers,
            tensor_idx,
            inputs_idx, 
            outputs_idx
        )
        
        # Store final network for contraction
        self.tensor_network_to_contract = identities + tensor_network_flat

    def measure(self, verbose: bool) -> np.ndarray:
        """Perform measurement on the tensor network to determine optimal task assignments.
        
        Measures each machine's tensor network node separately by:
        1. Replicating the full tensor network
        2. Connecting projection operators to all machines except the one being measured
        3. Contracting the network and finding the most probable task assignment
        
        Returns:
            np.ndarray: Array of measured task assignments for each machine
        """
        MAX_MEMORY = 2**31  # Limit to ~2GB memory
        result = np.zeros(self.num_machines, dtype=int)
        
        # Measure each machine separately
        for machine_idx in range(self.num_machines):
            if verbose:
                print('\rSolving variable ', machine_idx, ' of ', self.num_machines, end='')
            # Create fresh copies of tensors and projections
            tensors = tn.replicate_nodes(self.tensor_network_to_contract)
            projections = tn.replicate_nodes(self.projection)

            # Connect projections to all machines except current one
            for other_machine in range(self.num_machines):
                if other_machine != machine_idx:
                    tensors[other_machine][1] ^ projections[other_machine][0]

            # Setup network for contraction
            output_edge = [tensors[machine_idx][1]]
            nodes_to_contract = tensors + [
                projections[m] for m in range(self.num_machines) 
                if m != machine_idx
            ]
            
            # Contract network and find most probable assignment
            try:
                contracted = tn.contractors.auto(
                    nodes_to_contract,
                    output_edge_order=output_edge,
                    memory_limit=MAX_MEMORY
                )
                result[machine_idx] = np.argmax(abs(contracted.tensor))
            except MemoryError:
                if verbose:
                    print('Memory exceeded')
                result[machine_idx] = -1

        if verbose:
            print('\r                                       ', end='')
            print('\r', end='')

        return result

    def process(self, verbose: bool) -> np.ndarray:
        """Run full optimization process to find optimal task assignments.
        
        This method processes all constraint rules by iteratively:
        1. Taking rules for each machine
        2. Building tensor network layers from the rules
        3. Connecting layers and measuring final state
        
        Returns:
            np.ndarray: Optimized task assignments satisfying all constraints
        """
        # Process rules until none remain
        while any(self.working_constraints):
            # Each machine separately
            for rule_machine_set in self.working_constraints:
                if rule_machine_set:
                    # The rule that marks the condensed scheme to follow
                    basic_rule_scheme = rule_machine_set.pop(0)
                    layer, start_point_of_layer = self.rules_cycle(basic_rule_scheme)
                    self.total_tensors.append(layer)
                    self.start_of_layer.append(start_point_of_layer)

        # Connect all layers and measure final state
        self.connect_layers()
        return self.measure(verbose)

    def minimize(self, verbose: bool = False) -> np.ndarray:
        """Run global optimization process to find task assignments that minimize total processing time
        while satisfying all constraints.
        
        Returns:
            np.ndarray: Globally optimized task assignments mapped back to original machine ordering
        """
        # Try greedy solution first - assign each task to machine with minimum processing time
        greedy_solution = np.array([np.argmin(times) for times in self.costs_list_sorted], dtype=int)

        if self.no_rules:
            if verbose:
                print('Greedy solution for no constraints.')
            # If no constraints, greedy solution is optimal
            return greedy_solution

        # Check if greedy solution satisfies all constraints
        is_valid, _ = checker_with_classified_rules(self.constraints_sorted, greedy_solution)
        
        if is_valid:
            if verbose:
                print('Greedy solution.')
            # Map greedy solution back to original machine ordering
            return np.array([
                greedy_solution[self.old_new_machines[i]]
                for i in range(self.num_machines)
            ])

        # Greedy solution invalid - use tensor network optimization
        self.working_constraints = [rules.copy() for rules in self.constraints_sorted]
        tensor_solution = self.process(verbose)
        if -1 in tensor_solution:
            return np.ones(self.num_machines, dtype=int)*(-1)
        
        # Map tensor network solution back to original machine ordering
        return np.array([
            tensor_solution[self.old_new_machines[i]]
            for i in range(self.num_machines)
        ])


class IterativeOptimizer:
    """Iteratively optimizes task scheduling by progressively adding violated constraints.
    
    Starts with no constraints and adds constraints one by one as violations are found,
    along with any compatible constraints that can be condensed together.
    """
    def __init__(self, constraints: list[list[int|None]], costs_list: list[np.ndarray], tau: float):
        """Initialize the iterative optimizer.
        
        Args:
            constraints: Original list of all scheduling constraints
            costs_list: List of processing costs for each task on each machine
            tau: Time scaling factor for optimization
        """
        self.all_constraints = constraints
        self.costs_list = costs_list
        self.num_tasks = np.array([len(t) for t in self.costs_list], dtype=int)
        self.tau = tau
        
    def minimize(self, verbose: bool = False) -> np.ndarray:
        """Run iterative optimization process.
        
        Iteratively adds violated constraints and compatible ones that can be condensed together.
        Starts with no constraints and progressively builds up the constraint set as violations 
        are found, until either all constraints are satisfied or no better solution can be found.
        
        Args:
            verbose: If True, print optimization progress messages
            
        Returns:
            np.ndarray: Best found task assignments that either:
                       1. Satisfy all constraints
                       2. Are the last valid solution before failure
                       3. Are zeros if no valid solution found
        """
        # Start with empty constraint set
        current_constraints = [[], []]
        used_constraints = set()
        last_solution = None
        self.num_steps = 0
        self.num_constraints = 0
        
        try:
            while True:
                self.num_steps += 1
                # Optimize with current constraint set
                optimizer = Optimizer(current_constraints, self.costs_list, self.tau)
                solution = optimizer.minimize(verbose=False)
                if -1 in solution:
                    if verbose:
                        print('Memory exceeded')
                    return last_solution

                last_solution = solution
                
                # Check if solution satisfies all original constraints
                is_valid, violated_rule = checker(self.all_constraints, solution)
                
                if is_valid:
                    if verbose:
                        print(f"Found solution satisfying all constraints in {self.num_steps} steps with {len(used_constraints)} constraints")
                    return solution
                    
                # Skip if constraint already used
                if violated_rule in used_constraints:
                    if verbose:
                        print(f"All constraints used but violations remain in {self.num_steps} steps with {len(used_constraints)} constraints")
                    return last_solution
                
                # Add violated constraint and find compatible ones to condense
                used_constraints.add(violated_rule)
                new_conds = [self.all_constraints[0][violated_rule]]
                new_targets = [self.all_constraints[1][violated_rule]]
                
                # Get properties of violated constraint
                start_mach, final_mach, start_task, _ = start_end(new_conds[0])
                available_tasks = np.delete(np.arange(self.num_tasks[start_mach]), start_task)
                
                # Search for compatible constraints to condense
                for i, (cond, target) in enumerate(zip(self.all_constraints[0], self.all_constraints[1])):
                    # Skip if already used
                    if i in used_constraints:
                        continue
                        
                    # Check if constraint can be condensed
                    start_mach_new, final_mach_new, start_task_new, _ = start_end(cond)
                    
                    if (start_mach_new == start_mach and 
                        final_mach_new == final_mach and
                        start_task_new in available_tasks):
                        
                        new_conds.append(cond)
                        new_targets.append(target)
                        used_constraints.add(i)
                        available_tasks = np.delete(available_tasks, 
                            np.where(available_tasks == start_task_new)[0])
                            
                    # Stop if no more tasks available
                    if len(available_tasks) == 0:
                        break
                        
                # Add new constraints for next iteration
                current_constraints[0].extend(new_conds)
                current_constraints[1].extend(new_targets)

                self.num_constraints = len(used_constraints)
                    
        except Exception as e:
            if verbose:
                print(f"Error occurred: {str(e)}")
            # Return last valid solution or zeros if none found
            return last_solution if last_solution is not None else np.zeros(len(self.costs_list), dtype=int)



def basic_use_brute_force(constraints: list[list[int|None]], costs_list: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Run brute force optimization for task scheduling.
    
    Tries all possible task assignments to find the globally optimal solution that satisfies
    all constraints. Only practical for small problem sizes due to exponential complexity.
    
    Args:
        constraints: List of scheduling constraints to enforce
        costs_list: List of arrays containing processing costs for each task on each machine
        
    Returns:
        tuple:
            - np.ndarray: Optimal task schedule found by brute force
            - float: Total processing cost of optimal solution
    """
    best_cost, best_solution = brute_force(constraints, costs_list)

    print('\nBrute Force Solution:', best_solution)
    print('Total Processing Time:', best_cost)
    
    return best_solution, best_cost


def basic_use_tn(constraints: list[list[int|None]], costs_list: list[np.ndarray], tau: float) -> tuple[np.ndarray, float]:
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
    optimizer = Optimizer(constraints, costs_list, tau)
    result = optimizer.minimize(verbose=True)
    print('Tensor Network Solution:', result)

    # Validate solution and calculate total processing time
    if -1 in result:
        print('Memory exceeded')
        return result, np.inf

    is_valid, violations = checker(constraints, result)

    if is_valid:
        total_cost = sum(costs_list[i][int(result[i])] for i in range(len(costs_list)))
        print('Total Processing Time:', total_cost)
        return result, total_cost
    else:
        print('Solution violates scheduling constraints')
        print('Violations:', violations)
        return result, np.inf



def basic_use_iterative(constraints: list[list[int|None]], costs_list: list[np.ndarray], tau: float) -> tuple[np.ndarray, float]:
    """Run iterative optimization for task scheduling.
    
    Progressively adds violated constraints and optimizes using tensor networks.
    Starts with no constraints and adds them one by one as violations are found.
    
    Args:
        constraints: List of scheduling constraints to enforce
        costs_list: List of arrays containing processing costs for each task on each machine
        tau: Time scaling factor for imaginary time evolution
        
    Returns:
        tuple:
            - np.ndarray: Optimized task schedule 
            - float: Total processing cost (np.inf if solution invalid)
    """
    iterative_optimizer = IterativeOptimizer(constraints, costs_list, tau)
    result = iterative_optimizer.minimize(verbose=True)
    print('Iterative Optimizer Solution:', result)

    if -1 in result:
        print('Memory exceeded, solution:', result)
        return result, np.inf

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
