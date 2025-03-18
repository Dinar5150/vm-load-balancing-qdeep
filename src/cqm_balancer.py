# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dimod import BinaryQuadraticModel
from src.demo_enums import PriorityType
from dwave.samplers import SimulatedAnnealingSampler

# =============================================================================
# Helper: Penalty weights
# =============================================================================
# These weights can be tuned. In this example we enforce one-hot constraints
# strongly while encouraging (but not forcing) balanced resource loads.
ONE_HOT_WEIGHT = 1000  # weight for one-hot constraints (should be high)
LAMBDA_HARD = 1000     # weight for the prioritized resource (hard constraint)
LAMBDA_SOFT = 1        # weight for the non-prioritized resource (soft constraint)

# =============================================================================
# format_results
# =============================================================================
def format_results(
    plan: list[str], vms: dict, hosts: dict
) -> tuple[dict, dict]:
    """Transform a list of results into a dict pairing virtual machines with hosts.

    Args:
        plan: A list of strings where each string has the form ``vm_on_host``.
        vms: A dict of virtual machine dictionaries.
        hosts: A dict of host dictionaries.

    Returns:
        tuple[dict, dict]: The updated host dictionaries and 
        the updated virtual machine dictionaries.
    """
    for host in hosts:
        hosts[host]["cpu_used"] = 0
        hosts[host]["mem_used"] = 0

    for assignment in plan:
        vm_id, host_assignment = assignment.split("_on_")
        vm = vms[vm_id]

        hosts[host_assignment]["cpu_used"] += vm["cpu"]
        hosts[host_assignment]["mem_used"] += vm["mem"]

        vms[vm_id]["current_host"] = host_assignment

    return hosts, vms

# =============================================================================
# build_cqm -> build_bqm
# =============================================================================
def build_cqm(
    vms: dict, hosts: dict, priority: PriorityType
) -> BinaryQuadraticModel:
    """Define a QUBO formulation (as a BQM) for the balancing problem.

    This function builds a binary quadratic model by replacing the hard/soft
    inequality constraints of the original CQM with quadratic penalty terms.
    In particular, we add:
      - a one-hot penalty for each VM so that it is assigned to exactly one host;
      - for each host, quadratic penalties to encourage the load to match a target
        balanced value for CPU and memory.

    Args:
        vms: A dict of VM dicts containing current host and cpu and memory use.
        hosts: A dict of host dicts containing the CPU and memory cap as well as
            the current CPU and memory use.
        priority: Whether to prioritize balancing memory or CPU.

    Returns:
        BinaryQuadraticModel: The BQM model.
    """
    # Compute requested totals
    requested_cpu = {vm_id: vm_data["cpu"] for vm_id, vm_data in vms.items()}
    requested_mem = {vm_id: vm_data["mem"] for vm_id, vm_data in vms.items()}

    total_requested_cpu = sum(requested_cpu.values())
    total_requested_mem = sum(requested_mem.values())

    available_cpu = {host_id: host_data["cpu_cap"] for host_id, host_data in hosts.items()}
    available_mem = {host_id: host_data["mem_cap"] for host_id, host_data in hosts.items()}

    total_available_cpu = sum(available_cpu.values())
    total_available_mem = sum(available_mem.values())

    # Define target (balanced) load for each host (for CPU and mem)
    balanced_cpu = {
        host: available_cpu[host] * total_requested_cpu / total_available_cpu
        for host in available_cpu
    }
    balanced_mem = {
        host: available_mem[host] * total_requested_mem / total_available_mem
        for host in available_mem
    }

    # Choose penalty weights based on priority.
    # For the prioritized resource, we use a higher penalty (hard constraint).
    if priority is PriorityType.CPU:
        penalty_cpu = LAMBDA_HARD
        penalty_mem = LAMBDA_SOFT
    else:
        penalty_cpu = LAMBDA_SOFT
        penalty_mem = LAMBDA_HARD

    # We will build the QUBO (BQM) by accumulating linear and quadratic terms.
    linear = {}    # dict: variable -> coefficient
    quadratic = {} # dict: (var_i, var_j) -> coefficient
    offset = 0.0

    # Create variable names: each decision variable is named "vm_on_host".
    # We assume each vm must be assigned to one host.
    vm_list = list(vms.keys())
    host_list = list(hosts.keys())
    # Ensure every variable appears in the linear dictionary
    for vm in vm_list:
        for host in host_list:
            var = f"{vm}_on_{host}"
            linear[var] = 0.0

    # -----------------------------------------------------------------------------
    # Add one-hot (assignment) constraints for each vm:
    # For each vm, enforce (sum_{host in H} x_{vm,host} - 1)^2.
    # Expanded: 2*sum_{h<h'} x_{vm,h}x_{vm,h'} - sum_{h} x_{vm,h} + constant.
    # -----------------------------------------------------------------------------
    for vm in vm_list:
        # Linear and quadratic contributions for the one-hot term:
        # For each host, add -ONE_HOT_WEIGHT
        for host in host_list:
            var = f"{vm}_on_{host}"
            linear[var] += -ONE_HOT_WEIGHT
        # For each distinct pair of hosts, add +2*ONE_HOT_WEIGHT
        for i in range(len(host_list)):
            for j in range(i+1, len(host_list)):
                var_i = f"{vm}_on_{host_list[i]}"
                var_j = f"{vm}_on_{host_list[j]}"
                quadratic[(var_i, var_j)] = quadratic.get((var_i, var_j), 0.0) + 2 * ONE_HOT_WEIGHT
        # Constant term (ignored by the optimizer) would be ONE_HOT_WEIGHT, so we add it to offset.
        offset += ONE_HOT_WEIGHT

    # -----------------------------------------------------------------------------
    # Add resource balancing terms for each host.
    # For each host, we add a quadratic penalty term for CPU and for memory.
    # For CPU: penalty_cpu * ( (sum_{vm} cpu_vm * x_{vm,host}) - balanced_cpu[host] )^2.
    # Similarly for memory.
    # -----------------------------------------------------------------------------
    for host in host_list:
        # CPU terms:
        for idx, vm in enumerate(vm_list):
            var = f"{vm}_on_{host}"
            a = requested_cpu[vm]
            # Linear term from (a*x - target)²: a²*x - 2*target*a*x.
            linear[var] += penalty_cpu * (a * a - 2 * balanced_cpu[host] * a)
            # For each pair of distinct VMs, add the cross term.
            for jdx in range(idx+1, len(vm_list)):
                vm2 = vm_list[jdx]
                var2 = f"{vm2}_on_{host}"
                a2 = requested_cpu[vm2]
                quadratic[(var, var2)] = quadratic.get((var, var2), 0.0) + penalty_cpu * (2 * a * a2)
        # Add constant term for CPU (balanced_cpu[host]**2 * penalty_cpu) to offset.
        offset += penalty_cpu * (balanced_cpu[host] ** 2)

        # Memory terms:
        for idx, vm in enumerate(vm_list):
            var = f"{vm}_on_{host}"
            b = requested_mem[vm]
            linear[var] += penalty_mem * (b * b - 2 * balanced_mem[host] * b)
            for jdx in range(idx+1, len(vm_list)):
                vm2 = vm_list[jdx]
                var2 = f"{vm2}_on_{host}"
                b2 = requested_mem[vm2]
                quadratic[(var, var2)] = quadratic.get((var, var2), 0.0) + penalty_mem * (2 * b * b2)
        offset += penalty_mem * (balanced_mem[host] ** 2)

    # -----------------------------------------------------------------------------
    # Build the Binary Quadratic Model
    # -----------------------------------------------------------------------------
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")
    return bqm

# =============================================================================
# get_solution
# =============================================================================
def get_solution(bqm: BinaryQuadraticModel, time_limit: int) -> list[str]:
    """Call SA solver on the BQM and format results.

    Args:
        bqm: The Binary Quadratic Model.
        time_limit: Time limit (in seconds) to run the problem for.
            (Here used to set the number of sweeps in simulated annealing.)

    Returns:
        list[str]: A list of strings pairing virtual machines and hosts.
    """
    # Map time_limit (seconds) to a number of sweeps.
    # (This mapping can be adjusted according to your hardware/performance.)
    num_sweeps = time_limit * 100
    sampler = SimulatedAnnealingSampler()
    # Use a modest number of reads (you may increase num_reads as desired)
    sampleset = sampler.sample(bqm, num_reads=100, num_sweeps=num_sweeps)
    # Select the best (lowest energy) sample.
    best_sample = sampleset.first.sample
    # Return the list of variable names set to 1.
    result = [var for var, val in best_sample.items() if val == 1]
    return result
