import streamlit as st
import numpy as np
import pandas as pd
import json
from time import perf_counter
from optimizer import IterativeOptimizer
from genetic_algorithm import genetic_optimizer
from auxiliar_functions import create_instance, checker_sum

st.set_page_config(page_title="Task Scheduling Tensor Network Optimizer 🗓️", layout="wide")

st.title("Task Scheduling Tensor Network Optimizer 🗓️")
st.markdown("""
    This is an implementation of the research paper ["Task Scheduling Optimization from a Tensor Network Perspective"](https://arxiv.org/abs/2311.10433).
    
    The code was developed by **Alejandro Mata Ali** and is available on GitHub:
    [Task Scheduler with Tensor Networks Repository](https://github.com/DOKOS-TAYOS/Task_Scheduler_with_Tensor_Networks)
""")

# Description from paper
with st.expander("ℹ️ About Task Scheduling Optimizer and Usage Instructions", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## Problem Description 📝

        The task scheduling problem is a critical challenge in modern manufacturing and industrial settings. To understand the problem, take the example of the image.
        """)
        st.image("Machines.png")
        st.markdown("""
        Each machine can do a variety of tasks with its corresponding runtime, cutting in 10 seconds, painting in 3.7 seconds, or measure in 1.5 seconds.

        However, for certain reasons (constraints due to historical data or performance requirements), if the first machine cuts and the second rotates, the fourth must stack.

        The main problem is to obtain the optimal combination of tasks per machine that reduces the total runtime while satisfies the constraints.

        This is the problem this tool solves with the quantum-inspired tensor networks algorithm. This algorithm creates the tensor network representation of a tensor which provides the solution of the problem.

        There are two versions: the iterative one, which adds only the neccessary constraints until obtains the correct solution, and the genetic one, which applies a genetic algorithm to the iterative tensor network solver.
        
        ### Real-World Applications 🏭

        - **Manufacturing Plants**: Optimizing production lines to reduce bottlenecks 🏗️
        - **Electronics Assembly**: Coordinating component placement machines 🔌
        - **Food Processing**: Managing multiple processing and packaging lines 🥫
        - **Logistics Centers**: Organizing sorting and distribution tasks 📦
        """)

    with col2:
        st.markdown("""
        ### Mathematical Formulation 📊

        - **Variables**: 
            - $m$ machines (e.g., robotic arms, assembly stations)
            - Each machine $i$ has $P_i$ possible tasks
            - Processing time matrix $T_{ij}$ for task $j$ on machine $i$
            - Binary variable $x_{i}$ indicates which task is assigned to machine $i$

        - **Objective**: Minimize total production time while maintaining quality requirements ⚡

            Cost function: $C(\\vec{x})=\\sum_{i=0}^{m-1}T_{i,x_i}$
        - **Constraints**: Ensures related tasks are processed together for optimal results ⛓️


        ### How to Use This Tool 🛠️

        1. **Setup Your System** 🔧:
            - Enter the number of machines and tasks
            - Input processing times or generate random data
            - Define task constraints (which tasks must run together)

        2. **Choose Optimization Method** 🎯:
            - **Iterative Method**: Recommended for most cases
                - Faster resolution ⚡
                - Better for complex constraints 🔄
                - More reliable solutions ✅
            
            - **Genetic Algorithm**: Alternative approach
                - Joins the iterative algorithm with genetic algorithms 🔍
                - Different methodology 💡
                - Generally slower ⏳

        3. **Analyze Results** 📈:
            - Review task assignments 📋
            - Check constraint satisfaction ✔️
            - Examine total processing time ⏱️
            - Visualize the solution 📊
        """)
    st.markdown("The tool provides an intuitive interface to solve complex industrial scheduling problems that would be time-consuming to solve manually. 🚀")
st.markdown("---")
# Sidebar inputs
with st.sidebar:
    method = st.selectbox(
        "Optimization Method",
        ["Iterative Tensor Network Algorithm", "Genetic Tensor Network Algorithm"],
        index=0,
        help="Iterative method is recommended. Genetic algorithm generally gives worse solutions."
    )

    st.session_state.tau = st.slider(
        "Tau Parameter", 
        min_value=0.1,
        max_value=200.0, 
        value=100.0,
        step=0.1,
        help="Controls optimization approximation. Higher values approximate better the optimal solution, but can overflow."
    )
    
    st.session_state.num_machines = st.number_input("Number of Machines", 4, 10, 6)
    st.session_state.tasks_per_machine = st.number_input("Tasks per Machine", 4, 10, 6)

    if method == "Genetic Tensor Network Algorithm":
        st.warning("⚠️ Warning: Genetic algorithm typically provides worse solutions")
        
        st.subheader("Genetic Algorithm Parameters")
        ga_params = {
            "num_individuals": (10, 30, 20, "Number of Individuals"),
            "num_generations": (10, 30, 20, "Number of Generations"), 
            "max_constraints": (1, 10, 5, "Max Number of Constraints"),
            "tasks_per_individual": (2, 7, 3, "Tasks per Individual"),
            "num_crosses": (1, 10, 3, "Number of Crosses"),
            "num_mutations": (1, 10, 3, "Number of Mutations"),
            "proportion_parents": (2, 10, 3, "Proportion of Parents (1/proportion)")
        }
        
        for param, (min_val, max_val, default, label) in ga_params.items():
            st.session_state[param] = st.number_input(label, min_val, max_val, default)

# Main content
st.subheader("Task Processing Times ⏱️")

if st.button("Generate Random Instance Times 🎲"):
    st.session_state.processing_times = create_instance(1, st.session_state.num_machines, st.session_state.tasks_per_machine)[0]
    st.session_state.data = {
        'Machine': [f'Machine {i+1}' for i in range(st.session_state.num_machines)],
        **{f'Task {j+1}': [float(st.session_state.processing_times[i][j]) for i in range(st.session_state.num_machines)]
           for j in range(st.session_state.tasks_per_machine)}
    }

# Initialize processing times matrix if not exists
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = [np.zeros(st.session_state.tasks_per_machine) for _ in range(st.session_state.num_machines)]
    st.session_state.data = {
        'Machine': [f'Machine {i+1}' for i in range(st.session_state.num_machines)],
        **{f'Task {j+1}': [float(st.session_state.processing_times[i][j]) for i in range(st.session_state.num_machines)]
           for j in range(st.session_state.tasks_per_machine)}
    }

# Handle dimension changes
current_machines = len(st.session_state.processing_times)
current_tasks = len(st.session_state.processing_times[0])
target_machines = st.session_state.num_machines
target_tasks = st.session_state.tasks_per_machine

if current_machines != target_machines or current_tasks != target_tasks:
    # Clear constraints
    st.session_state.constraints = []
    st.session_state.targets = []
    # Adjust number of machines
    if current_machines < target_machines:
        st.session_state.processing_times.extend([np.zeros(target_tasks) for _ in range(target_machines - current_machines)])
    elif current_machines > target_machines:
        st.session_state.processing_times = st.session_state.processing_times[:target_machines]
    
    # Adjust number of tasks per machine
    if current_tasks < target_tasks:
        for i in range(target_machines):
            st.session_state.processing_times[i] = np.pad(st.session_state.processing_times[i], (0, target_tasks - current_tasks))
    elif current_tasks > target_tasks:
        for i in range(target_machines):
            st.session_state.processing_times[i] = st.session_state.processing_times[i][:target_tasks]
    
    # Update data dictionary
    st.session_state.data = {
        'Machine': [f'Machine {i+1}' for i in range(target_machines)],
        **{f'Task {j+1}': [float(st.session_state.processing_times[i][j]) for i in range(target_machines)]
           for j in range(target_tasks)}
    }

# Create and display editable dataframe
df = pd.DataFrame(st.session_state.data)
edited_df = st.data_editor(
    df,
    key='processing_times_editor',
    column_config={
        'Machine': st.column_config.TextColumn('Machine', disabled=True),
        **{f'Task {j+1}': st.column_config.NumberColumn(f'Task {j+1}', min_value=0.0)
           for j in range(st.session_state.tasks_per_machine)}
    }
)

# Update processing times from edited dataframe
for i in range(st.session_state.num_machines):
    st.session_state.processing_times[i] = np.array([
        edited_df.iloc[i][f'Task {j+1}'] for j in range(st.session_state.tasks_per_machine)
    ])


# Constraints input
st.markdown('---')
st.subheader("Constraints 🔗")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    Add scheduling constraints that specify which tasks must be processed together. 🔄

    **Format** 📝:
    - **Condition**: List of task indices or None for each machine (e.g. `[1,None,3,None]`)
    - **Target**: Machine and task to be scheduled `[machine_idx, task_idx]`
    """)

with col2:
    st.markdown("""
    **Example** 💡: `[1,None,3,None], [1,2]` means:
    - If Task 1 from machine 0 and task 3 from machine 2 are processed, then, task 2 must be scheduled on machine 1.
    - Machine 3 does not condition the machine 1.
    """)

if 'constraints' not in st.session_state:
    st.session_state.constraints = []
    st.session_state.targets = []

# Split into two columns
button_col, display_col = st.columns(2)

with button_col:
    # Random constraint generation
    num_constraints = st.number_input(
        "Number of Random Constraints",
        min_value=0,
        max_value=10,
        value=2,
        help="Generate random valid constraints"
    )
    
    if st.button("Generate Random Constraints 🎲", use_container_width=True):
        instance = create_instance(
            num_constraints, 
            st.session_state.num_machines,
            st.session_state.tasks_per_machine
        )[1]
        st.session_state.constraints = instance[0]
        st.session_state.targets = instance[1]

    # Manual constraint input
    col1, col2 = st.columns(2)
    with col1:
        condition = st.text_input(
            "Condition", 
            help="Task indices (comma-separated). Use 'None' for machines not involved."
        )
    with col2:
        target = st.text_input(
            "Target",
            help="Machine index and task index (comma-separated)"
        )

    if st.button("Add Constraint ➕", use_container_width=True):
        try:
            # Parse condition input
            cond = [
                None if x.strip().lower() == 'none' else int(x) 
                for x in condition.split(',')
            ]
            
            # Validate condition length
            if len(cond) != st.session_state.num_machines:
                st.error(f"Condition must specify {st.session_state.num_machines} values")
                raise ValueError()
                
            # Parse target input
            targ = [int(x) for x in target.split(',')]
            
            # Validate target
            if len(targ) != 2:
                st.error("Target must contain machine index and task index")
                raise ValueError()
            
            if targ[0] >= st.session_state.num_machines:
                st.error("Invalid machine index in target")
                raise ValueError()
                
            if targ[1] >= st.session_state.tasks_per_machine:
                st.error("Invalid task index in target") 
                raise ValueError()
            
            # Check for duplicate constraint
            if any(c == cond and t == targ 
                  for c, t in zip(st.session_state.constraints, st.session_state.targets)):
                st.error("This constraint already exists")
            else:
                st.session_state.constraints.append(cond)
                st.session_state.targets.append(targ)
                st.success("Constraint added successfully! ✨")
        except ValueError:
            pass
        except:
            st.error("Invalid input format")

with display_col:
    # Display existing constraints
    if st.session_state.constraints:
        if st.button("Clear All Constraints 🗑️", use_container_width=True):
            st.session_state.constraints = []
            st.session_state.targets = []
            st.rerun()
            
        st.write("Current Constraints:")
        for i, (cond, target) in enumerate(zip(st.session_state.constraints, st.session_state.targets)):
            with st.expander(f"Constraint {i+1}: {cond}, {target}"):
                
                # Generate description
                condition_desc = []
                for machine, task in enumerate(cond):
                    if task is not None:
                        condition_desc.append(f"machine {machine} runs task {task},")
                
                if not condition_desc:
                    condition_text = "No specific task assignments required"
                else:
                    condition_text = "When " + " and ".join(condition_desc)
                    
                target_text = f"Then machine {target[0]} should run task {target[1]}."
                
                st.write("**Description** 📝:")
                st.write(condition_text)
                st.write(target_text)
                
                if st.button(f"Remove 🗑️", key=f"remove_{i}"):
                    st.session_state.constraints.pop(i)
                    st.session_state.targets.pop(i)
                    st.success("Constraint removed ✨")
                    st.rerun()


# Optimize button
st.markdown('---')
if st.button("Optimize! 🚀", use_container_width=True):
    constraints = [st.session_state.constraints, st.session_state.targets]
    
    with st.spinner("Optimizing... ⚙️"):
        try:
            start_time = perf_counter()
            if method == "Iterative Tensor Network Algorithm":
                optimizer = IterativeOptimizer(constraints, st.session_state.processing_times, st.session_state.tau)
                solution = optimizer.minimize(verbose=False)
            else:
                solution = genetic_optimizer(
                    constraints,
                    st.session_state.processing_times,
                    st.session_state.tau,
                    st.session_state.num_individuals,
                    st.session_state.num_generations,
                    st.session_state.max_constraints,
                    st.session_state.tasks_per_individual,
                    st.session_state.num_crosses,
                    st.session_state.num_mutations,
                    iterative=True,
                    parent_proportion=st.session_state.proportion_parents,
                    verbose=False
                )[0]
            runtime = perf_counter() - start_time
                
            # Display results
            st.subheader("Results 📊")
            
            if -1 in solution:
                st.error("Memory exceeded during optimization 💾")
            else:
                is_valid, num_not_satisfied = checker_sum(constraints, solution)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Constraint Status")
                    if is_valid:
                        st.success("✅ All constraints are satisfied!", icon="✨")
                    else:
                        st.error(f"❌ {num_not_satisfied} constraints not satisfied", icon="⚠️")
                        
                    if method == "Iterative Tensor Network Algorithm":
                        st.info(f"🔄 Number of iterations: {optimizer.num_steps}", icon="🔄")
                    st.info(f"⏱️ Algorithm runtime: {runtime:.3e} seconds", icon="⏱️")
                with col2:
                    st.markdown("### ⌛ Total Time")
                    total_cost = sum(st.session_state.processing_times[i][int(solution[i])] for i in range(len(st.session_state.processing_times)))
                    st.markdown(f"""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        background-color: #2d2d2d;
                        text-align: center;
                        font-size: 1.2em;
                        color: #ffffff;
                    '>
                        <span style='color: #ffffff'>⏰ Total Processing Time:</span><br>
                        <span style='font-size: 2em; font-weight: bold; color: #ffffff'>{total_cost:.3f}s</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 📋 Task Assignments")
                ROW_AMOUNT = len(solution) // 2 + len(solution) % 2
                for row in range(2):
                    cols = st.columns(ROW_AMOUNT)
                    for i in range(ROW_AMOUNT):
                        idx = row * ROW_AMOUNT + i
                        if idx < len(solution):
                            with cols[i]:
                                st.markdown(f"""
                                <div style='
                                    padding: 10px;
                                    border-radius: 5px;
                                    background-color: #1e1e1e;
                                    margin: 5px 0;
                                    font-size: 0.8em;
                                    color: #ffffff;
                                '>
                                    <span style='color: #ffffff'>🔧 Machine {idx+1} → 📝 Task {solution[idx]+1}<br>
                                    ⏱️ Time: {st.session_state.processing_times[idx][int(solution[idx])]:.4f}s</span>
                                </div>
                                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Solution Visualization")
                ROW_AMOUNT = 10
                num_rows = int(np.ceil(st.session_state.num_machines/ROW_AMOUNT))
                var_cols = []
                for row in range(num_rows):
                    var_cols.append(st.columns(len(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)])))
                    # Create boxes for each variable
                    for i, val in enumerate(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)]):
                        with var_cols[row][i]:
                            # Start variable container
                            st.markdown(f"""
                                <div style='
                                    width: 100%;
                                    text-align: center;
                                    display: flex;
                                    flex-direction: row;
                                    align-items: center;
                                    justify-content: center;
                                    background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
                                    padding: 8px;
                                    border-radius: 8px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                    border: 1px solid #3a3a3a;
                                    margin-bottom: 4px;
                                '>
                                    <div style='
                                        font-weight: bold;
                                        margin-right: 8px;
                                        font-size: 0.9em;
                                        color: #e0e0e0;
                                    '>🔧 Machine {ROW_AMOUNT*row+i+1}</div>
                                    <div style='
                                        display: flex;
                                        flex-direction: row;
                                        gap: 0px;
                                        flex: 1;
                                        justify-content: center;
                                    '>
                            """, unsafe_allow_html=True)
                            
                            # Create boxes for each possible value
                            for d in range(st.session_state.tasks_per_machine):
                                background = "#ff4d4d" if d == val else "#4d79ff"
                                border_color = "#8b0000" if d == val else "#00008b"
                                hover_bg = "#ff3333" if d == val else "#3366ff"
                                runtime = st.session_state.processing_times[ROW_AMOUNT*row+i][d]
                                st.markdown(f"""
                                    <div style='
                                        background: {background};
                                        color: white;
                                        border: 2px solid {border_color};
                                        padding: 6px;
                                        margin: 0;
                                        font-size: 0.9em;
                                        flex: 1;
                                        text-align: center;
                                        transition: all 0.2s ease;
                                        box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
                                        cursor: pointer;
                                        font-weight: 500;
                                        border-radius: 8px;
                                    '>
                                        <div style='margin-bottom: 2px;'>📝 Task {d+1}</div>
                                        <div style='font-size: 0.8em; opacity: 0.9;'>⏱️ {runtime:.2f}s</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                
                # Create download button for results
                results = {
                    "solution": solution.tolist(),
                    "total_cost": float(total_cost),
                    "processing_times": [t.tolist() for t in st.session_state.processing_times],
                    "constraints": st.session_state.constraints,
                    "targets": st.session_state.targets,
                    "constraints_satisfied": is_valid,
                    "number_not_satisfied": max(num_not_satisfied,0),
                }
                st.download_button(
                    "Download Results 📥",
                    data=json.dumps(results, indent=2),
                    file_name="task_scheduling_results.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")