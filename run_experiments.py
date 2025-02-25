import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.simulation import HospitalSimulation
from src.patient import Patient

# Fix for the KeyError: 'General' in analyze_scalability function
def analyze_scalability(config_file, max_departments=3):
    """
    Analyze simulation scalability with increasing departments
    """
    print("Running scalability analysis...")
    results = []
    
    # Load configuration to check available departments
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get actual number of departments available
    available_depts = len(config['departments'])
    max_departments = min(max_departments, available_depts)
    
    for n_dept in range(1, max_departments + 1):
        sim = HospitalSimulation(config_file)
        
        # Modified to safely set departments
        sim.set_departments(n_dept)
        
        start_time = time.time()
        sim_results = sim.run_simulation(duration=240)  # Shorter duration for experiments
        execution_time = time.time() - start_time
        
        results.append({
            'departments': n_dept,
            'execution_time': execution_time,
            'throughput': len(sim.completed_patients),
            'avg_wait_time': sim_results['overall_metrics']['avg_wait_time'] if 'avg_wait_time' in sim_results['overall_metrics'] else 0
        })
        
        print(f"  Completed {n_dept} departments: {execution_time:.2f} seconds, {len(sim.completed_patients)} patients")
        
    return pd.DataFrame(results)

# Fix for the set_departments method in HospitalSimulation class
def set_departments(self, num_departments):
    """
    Helper method to adjust the number of departments for scalability testing
    """
    # Get list of available department names
    dept_names = list(self.config['departments'].keys())
    
    if num_departments <= 0 or num_departments > len(dept_names):
        print(f"Warning: Requested {num_departments} departments, but only {len(dept_names)} are available")
        num_departments = min(max(1, num_departments), len(dept_names))
    
    selected_depts = dept_names[:num_departments]
    print(f"Using departments: {selected_depts}")
    
    # Keep only the selected departments
    self.departments = {name: self.departments[name] for name in selected_depts}
    
    # Important: Update patient journey to use only available departments
    if "ER" not in selected_depts and len(selected_depts) > 0:
        # If ER is not available, use the first department as entry point
        self.entry_department = selected_depts[0]
    else:
        self.entry_department = "ER"

# Fix for patient_journey method in HospitalSimulation class
def patient_journey(self, patient, initial_department):
    """
    Model the complete journey of a patient through the hospital
    """
    # Check if the initial department exists
    if initial_department not in self.departments:
        # Fallback to the first available department
        if not self.departments:
            # No departments available, discharge patient immediately
            patient.status = "discharged"
            self.completed_patients.append(patient)
            return
        initial_department = list(self.departments.keys())[0]
    
    current_dept = initial_department
    
    while True:
        # Process treatment in current department
        dept = self.departments[current_dept]
        yield self.env.process(dept.treat_patient(patient))
        
        # Evaluate if transfer is needed
        next_dept, transfer_prob = dept.evaluate_transfer(patient)
        
        # Only transfer if the next department exists
        if next_dept and next_dept in self.departments and random.random() < transfer_prob:
            # Handle transfer
            transfer_time = self.env.now
            patient.add_transfer(current_dept, next_dept, transfer_time)
            dept.patients_transferred += 1
            self.monitor.record_transfer(patient, current_dept, next_dept, transfer_time)
            
            # Move to next department
            current_dept = next_dept
        else:
            # Patient journey complete
            break
            
    # Patient is discharged
    patient.status = "discharged"
    self.completed_patients.append(patient)
    
    # Calculate and record total wait time
    total_wait = patient.get_total_wait_time()
    self.monitor.record_wait_time(patient, total_wait, current_dept)

# Fix for generate_patient method in HospitalSimulation class
def generate_patient(self, rate=0.1):
    """
    Generate patients arriving at the hospital
    Returns a generator for the simulation environment
    """
    # Possible patient conditions
    conditions = ["cardiac", "trauma", "respiratory", "general"]
    condition_weights = [0.2, 0.3, 0.25, 0.25]  # Probability distribution
    
    # Define an entry department (default to ER if available)
    entry_department = getattr(self, 'entry_department', "ER")
    if entry_department not in self.departments and self.departments:
        entry_department = list(self.departments.keys())[0]
    
    while True:
        # Check if there are any departments
        if not self.departments:
            # No departments available, stop generating patients
            return
            
        # Generate inter-arrival time (exponential distribution)
        interarrival = np.random.exponential(1/rate)
        yield self.env.timeout(interarrival)
        
        # Create new patient
        self.patient_counter += 1
        condition = np.random.choice(conditions, p=condition_weights)
        patient = Patient(self.patient_counter, condition)
        patient.arrival_time = self.env.now
        self.patients.append(patient)
        
        # Start patient journey with the appropriate entry department
        self.env.process(self.patient_journey(patient, entry_department))

# Fixed Department get_statistics method to handle empty wait_times
def get_statistics(self):
    """Return department performance statistics"""
    # Handle case with no wait times
    if not self.wait_times:
        return {
            'department': self.name,
            'patients_treated': self.patients_treated,
            'patients_transferred': self.patients_transferred,
            'avg_wait_time': 0,
            'max_wait_time': 0,
            'avg_bed_utilization': self.beds.count / max(1, self.beds.capacity) if hasattr(self, 'beds') else 0,
            'avg_staff_utilization': self.staff.count / max(1, self.staff.capacity) if hasattr(self, 'staff') else 0
        }
    
    # Calculate wait time statistics
    try:
        avg_wait = sum(w['wait_time'] for w in self.wait_times) / max(1, len(self.wait_times))
        max_wait = max(w['wait_time'] for w in self.wait_times) if self.wait_times else 0
    except (KeyError, TypeError):
        # Handle case where wait_times might have invalid format
        avg_wait = 0
        max_wait = 0
    
    # Handle case with no occupancy history
    if not self.occupancy_history:
        return {
            'department': self.name,
            'patients_treated': self.patients_treated,
            'patients_transferred': self.patients_transferred,
            'avg_wait_time': avg_wait,
            'max_wait_time': max_wait,
            'avg_bed_utilization': self.beds.count / max(1, self.beds.capacity) if hasattr(self, 'beds') else 0,
            'avg_staff_utilization': self.staff.count / max(1, self.staff.capacity) if hasattr(self, 'staff') else 0
        }
    
    # Calculate occupancy statistics
    try:
        avg_bed_util = sum(o.get('bed_utilization', 0) for o in self.occupancy_history) / max(1, len(self.occupancy_history))
        avg_staff_util = sum(o.get('staff_utilization', 0) for o in self.occupancy_history) / max(1, len(self.occupancy_history))
    except (KeyError, TypeError):
        # Handle case where occupancy_history might have invalid format
        avg_bed_util = 0
        avg_staff_util = 0
    
    return {
        'department': self.name,
        'patients_treated': self.patients_treated,
        'patients_transferred': self.patients_transferred,
        'avg_wait_time': avg_wait,
        'max_wait_time': max_wait,
        'avg_bed_utilization': avg_bed_util,
        'avg_staff_utilization': avg_staff_util
    }

def optimize_resources(config_file, department="ER"):
    """
    Experiment with different resource allocations
    """
    print(f"Optimizing resources for {department}...")
    results = []
    
    # Load original config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create temporary config file for experiments
    temp_config_path = "temp_config.json"
    
    for beds in range(5, 31, 5):
        for staff in range(5, 26, 5):
            # Modify config
            config['departments'][department]['beds'] = beds
            config['departments'][department]['staff'] = staff
            
            # Save modified config
            with open(temp_config_path, 'w') as f:
                json.dump(config, f)
                
            # Run simulation with modified config
            sim = HospitalSimulation(temp_config_path)
            metrics = sim.run_and_analyze()
            
            results.append({
                'beds': beds,
                'staff': staff,
                'wait_time': metrics['avg_wait_time'],
                'utilization': metrics['resource_utilization'],
                'throughput': metrics['throughput']
            })
            
            print(f"  Beds: {beds}, Staff: {staff}, Wait Time: {metrics['avg_wait_time']:.2f}, Util: {metrics['resource_utilization']:.2f}")
    
    # Clean up temp file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    return pd.DataFrame(results)

def compare_patient_loads(config_file):
    """
    Compare system performance under different patient loads
    """
    print("Comparing different patient loads...")
    results = []
    
    arrival_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    for rate in arrival_rates:
        sim = HospitalSimulation(config_file)
        sim_results = sim.run_simulation(patient_arrival_rate=rate)
        
        results.append({
            'arrival_rate': rate,
            'patients_per_hour': rate * 60,
            'completed_patients': sim_results['completed_patients'],
            'avg_wait_time': sim_results['overall_metrics']['avg_wait_time'],
            'transfer_rate': sim_results['overall_metrics']['transfer_rate'],
            'completion_rate': sim_results['completed_patients'] / sim_results['total_patients']
        })
        
        print(f"  Rate: {rate}, Completed: {sim_results['completed_patients']}, Wait: {sim_results['overall_metrics']['avg_wait_time']:.2f}")
        
    return pd.DataFrame(results)

def visualize_results(scalability_results, optimization_results, load_results):
    """
    Create visualizations of experiment results
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Scalability Analysis
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=scalability_results, x="departments", y="execution_time", marker='o')
    plt.title("Execution Time vs. Number of Departments")
    plt.xlabel("Number of Departments")
    plt.ylabel("Execution Time (seconds)")
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=scalability_results, x="departments", y="throughput", marker='o')
    plt.title("Patient Throughput vs. Number of Departments")
    plt.xlabel("Number of Departments")
    plt.ylabel("Patients Processed")
    
    plt.tight_layout()
    plt.savefig("results/scalability_analysis.png")
    
    # 2. Resource Optimization
    plt.figure(figsize=(15, 10))
    
    # Create a pivot table for heatmaps
    wait_pivot = optimization_results.pivot("beds", "staff", "wait_time")
    util_pivot = optimization_results.pivot("beds", "staff", "utilization")
    throughput_pivot = optimization_results.pivot("beds", "staff", "throughput")
    
    plt.subplot(2, 2, 1)
    sns.heatmap(wait_pivot, annot=True, fmt=".1f", cmap="YlGnBu_r")
    plt.title("Average Wait Time by Resource Allocation")
    plt.xlabel("Staff")
    plt.ylabel("Beds")
    
    plt.subplot(2, 2, 2)
    sns.heatmap(util_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Resource Utilization by Allocation")
    plt.xlabel("Staff")
    plt.ylabel("Beds")
    
    plt.subplot(2, 2, 3)
    sns.heatmap(throughput_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Patient Throughput by Resource Allocation")
    plt.xlabel("Staff")
    plt.ylabel("Beds")
    
    # Find and mark optimal configuration
    # (This is a simplified approach - in reality, you'd use a more sophisticated optimization metric)
    optimal_config = optimization_results.iloc[
        optimization_results['wait_time'].idxmin()
    ]
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=optimization_results, x="utilization", y="wait_time", 
                    size="throughput", sizes=(20, 200), hue="beds", palette="viridis")
    plt.title("Wait Time vs. Utilization")
    plt.xlabel("Resource Utilization")
    plt.ylabel("Average Wait Time")
    
    plt.tight_layout()
    plt.savefig("results/resource_optimization.png")
    
    # 3. Patient Load Comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.lineplot(data=load_results, x="patients_per_hour", y="avg_wait_time", marker='o')
    plt.title("Wait Time vs. Patient Arrival Rate")
    plt.xlabel("Patients per Hour")
    plt.ylabel("Average Wait Time")
    
    plt.subplot(2, 2, 2)
    sns.lineplot(data=load_results, x="patients_per_hour", y="completed_patients", marker='o')
    plt.title("Completed Patients vs. Arrival Rate")
    plt.xlabel("Patients per Hour")
    plt.ylabel("Completed Patients")
    
    plt.subplot(2, 2, 3)
    sns.lineplot(data=load_results, x="patients_per_hour", y="transfer_rate", marker='o')
    plt.title("Transfer Rate vs. Patient Arrival Rate")
    plt.xlabel("Patients per Hour")
    plt.ylabel("Transfer Rate")
    
    plt.subplot(2, 2, 4)
    sns.lineplot(data=load_results, x="patients_per_hour", y="completion_rate", marker='o')
    plt.title("Completion Rate vs. Patient Arrival Rate")
    plt.xlabel("Patients per Hour")
    plt.ylabel("Completion Rate")
    
    plt.tight_layout()
    plt.savefig("results/patient_load_comparison.png")
    
    print("Visualizations saved to results directory")

def visualize_simulation_results(simulation_results):
    """
    Create visualizations for a single simulation run with improved error handling
    """
    # Create directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Convert to dataframes for easier plotting
    dept_stats = pd.DataFrame(simulation_results['department_stats'])
    patient_stats = pd.DataFrame(simulation_results['patient_stats']) if 'patient_stats' in simulation_results else pd.DataFrame()
    
    # Debug information
    print("Department Statistics DataFrame:")
    print(dept_stats)
    
    if not dept_stats.empty:
        print("\nAvailable columns in department_stats:")
        for col in dept_stats.columns:
            print(f"- {col}: {dept_stats[col].dtype}, {dept_stats[col].isna().sum()} NaN values")
            if col == 'avg_wait_time':
                print(f"  Values: {dept_stats[col].tolist()}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Department statistics
    plt.subplot(2, 3, 1)
    if not dept_stats.empty and 'department' in dept_stats.columns and 'avg_wait_time' in dept_stats.columns:
        # Ensure data is numeric and non-negative
        dept_stats['avg_wait_time'] = pd.to_numeric(dept_stats['avg_wait_time'], errors='coerce').fillna(0)
        dept_stats['avg_wait_time'] = dept_stats['avg_wait_time'].clip(lower=0)  # Clip negative values to 0
        
        # Plot
        ax = sns.barplot(x='department', y='avg_wait_time', data=dept_stats)
        plt.title('Average Wait Time by Department')
        plt.xticks(rotation=45)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       xytext=(0, 5), 
                       textcoords='offset points')
    else:
        plt.text(0.5, 0.5, 'No wait time data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Average Wait Time by Department (No Data)')
    
    plt.subplot(2, 3, 2)
    if not dept_stats.empty and 'department' in dept_stats.columns and 'avg_bed_utilization' in dept_stats.columns:
        # Ensure data is numeric
        dept_stats['avg_bed_utilization'] = pd.to_numeric(dept_stats['avg_bed_utilization'], errors='coerce').fillna(0)
        
        # Plot
        ax = sns.barplot(x='department', y='avg_bed_utilization', data=dept_stats)
        plt.title('Bed Utilization by Department')
        plt.xticks(rotation=45)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       xytext=(0, 5), 
                       textcoords='offset points')
    else:
        plt.text(0.5, 0.5, 'No utilization data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Bed Utilization by Department (No Data)')
    
    plt.subplot(2, 3, 3)
    if not dept_stats.empty and 'department' in dept_stats.columns and 'patients_treated' in dept_stats.columns:
        # Ensure data is numeric
        dept_stats['patients_treated'] = pd.to_numeric(dept_stats['patients_treated'], errors='coerce').fillna(0)
        
        # Plot
        ax = sns.barplot(x='department', y='patients_treated', data=dept_stats)
        plt.title('Patients Treated by Department')
        plt.xticks(rotation=45)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       xytext=(0, 5), 
                       textcoords='offset points')
    else:
        plt.text(0.5, 0.5, 'No patient treatment data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Patients Treated by Department (No Data)')
    
    # Patient statistics
    plt.subplot(2, 3, 4)
    if not patient_stats.empty and 'wait_time' in patient_stats.columns:
        # Ensure data is numeric and non-negative
        patient_stats['wait_time'] = pd.to_numeric(patient_stats['wait_time'], errors='coerce').fillna(0)
        patient_stats['wait_time'] = patient_stats['wait_time'].clip(lower=0)  # Clip negative values to 0
        
        sns.histplot(data=patient_stats, x='wait_time', kde=True)
        plt.title('Wait Time Distribution')
    else:
        plt.text(0.5, 0.5, 'No wait time distribution data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Wait Time Distribution (No Data)')
    
    plt.subplot(2, 3, 5)
    if not patient_stats.empty and 'treatment_time' in patient_stats.columns:
        # Ensure data is numeric and non-negative
        patient_stats['treatment_time'] = pd.to_numeric(patient_stats['treatment_time'], errors='coerce').fillna(0)
        patient_stats['treatment_time'] = patient_stats['treatment_time'].clip(lower=0)  # Clip negative values to 0
        
        sns.histplot(data=patient_stats, x='treatment_time', kde=True)
        plt.title('Treatment Time Distribution')
    else:
        plt.text(0.5, 0.5, 'No treatment time data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Treatment Time Distribution (No Data)')
    
    plt.subplot(2, 3, 6)
    if not patient_stats.empty and 'condition' in patient_stats.columns:
        ax = sns.countplot(data=patient_stats, x='condition')
        plt.title('Patient Conditions')
        plt.xticks(rotation=45)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       xytext=(0, 5), 
                       textcoords='offset points')
    else:
        plt.text(0.5, 0.5, 'No patient condition data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Patient Conditions (No Data)')
    
    plt.tight_layout()
    plt.savefig('results/simulation_results.png')
    print("Simulation visualizations saved to results directory")

# Add the run_and_analyze method to the HospitalSimulation class if it doesn't exist
def run_and_analyze(self, duration=720, patient_arrival_rate=0.1):
    """
    Run the simulation and analyze the results
    Returns a dictionary of metrics
    """
    # Run the simulation
    simulation_results = self.run_simulation(duration, patient_arrival_rate)
    
    # Extract key metrics
    total_patients = simulation_results['total_patients']
    completed_patients = simulation_results['completed_patients']
    
    # Calculate overall metrics
    avg_wait_time = simulation_results['overall_metrics']['avg_wait_time']
    
    # Calculate resource utilization
    department_stats = simulation_results['department_stats']
    avg_bed_utilization = sum(d['avg_bed_utilization'] for d in department_stats) / len(department_stats) if department_stats else 0
    avg_staff_utilization = sum(d['avg_staff_utilization'] for d in department_stats) / len(department_stats) if department_stats else 0
    
    # Average utilization
    resource_utilization = (avg_bed_utilization + avg_staff_utilization) / 2
    
    # Return summary metrics
    metrics = {
        'total_patients': total_patients,
        'completed_patients': completed_patients,
        'completion_rate': completed_patients / total_patients if total_patients > 0 else 0,
        'avg_wait_time': avg_wait_time,
        'resource_utilization': resource_utilization,
        'throughput': completed_patients / (duration / 60)  # Patients per hour
    }
    
    return {**metrics, **simulation_results}  # Combine with full simulation results

def main():
    """
    Main function to run all experiments
    """
    # Check if config.json exists, otherwise use a default path
    config_file = "config.json"
    if not os.path.exists(config_file):
        # Try to find in the current directory
        config_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if config_files:
            config_file = config_files[0]
            print(f"Using configuration file: {config_file}")
        else:
            print("Warning: No configuration file found. Please create a config.json file.")
            return
    
    print("Starting hospital simulation experiments...")
    
    # Add the required methods to HospitalSimulation class
    HospitalSimulation.set_departments = set_departments
    HospitalSimulation.patient_journey = patient_journey
    HospitalSimulation.generate_patient = generate_patient
    HospitalSimulation.run_and_analyze = run_and_analyze
    
    try:
        # Run scalability analysis
        scalability_results = analyze_scalability(config_file)
        
        # Run resource optimization
        optimization_results = optimize_resources(config_file)
        
        # Run patient load comparison
        load_results = compare_patient_loads(config_file)
        
        # Visualize all results
        visualize_results(scalability_results, optimization_results, load_results)
        
        # Run a single simulation with default parameters for detailed analysis
        sim = HospitalSimulation(config_file)
        sim_results = sim.run_and_analyze()
        visualize_simulation_results(sim_results)
        
        print("All experiments completed successfully.")
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()