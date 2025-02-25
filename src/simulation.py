import simpy
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from .patient import Patient
from .department import Department

class PerformanceMonitor:
    """
    Monitors and records simulation performance metrics
    """
    def __init__(self):
        self.wait_times = []
        self.occupancy_rates = defaultdict(list)
        self.transfer_rates = []
        self.critical_events = []
        self.patient_flow = defaultdict(int)
        
    def record_wait_time(self, patient, duration, department):
        self.wait_times.append({
            'patient_id': patient.id,
            'condition': patient.condition,
            'department': department,
            'duration': duration
        })
        
    def record_occupancy(self, department, time, rate):
        self.occupancy_rates[department].append({
            'time': time,
            'rate': rate
        })
        
    def record_transfer(self, patient, from_dept, to_dept, time):
        self.transfer_rates.append({
            'patient_id': patient.id,
            'condition': patient.condition,
            'from_dept': from_dept,
            'to_dept': to_dept,
            'time': time
        })
        
    def record_patient_flow(self, time, department, count):
        self.patient_flow[(time, department)] = count
        
    def record_critical_event(self, event_type, time, details):
        self.critical_events.append({
            'event_type': event_type,
            'time': time,
            'details': details
        })
        
    def calculate_metrics(self):
        if not self.wait_times:
            return {
                'avg_wait_time': 0,
                'max_wait_time': 0,
                'occupancy_rate': 0,
                'transfer_rate': 0
            }
            
        wait_times_df = pd.DataFrame(self.wait_times)
        transfers_df = pd.DataFrame(self.transfer_rates) if self.transfer_rates else pd.DataFrame()
        
        metrics = {
            'avg_wait_time': wait_times_df['duration'].mean(),
            'max_wait_time': wait_times_df['duration'].max(),
            'occupancy_rate': np.mean([item['rate'] for dept in self.occupancy_rates for item in self.occupancy_rates[dept]]),
            'transfer_rate': len(self.transfer_rates) / len(self.wait_times) if self.wait_times else 0
        }
        
        # Add department-specific metrics
        if not wait_times_df.empty and 'department' in wait_times_df.columns:
            for dept in wait_times_df['department'].unique():
                dept_waits = wait_times_df[wait_times_df['department'] == dept]
                metrics[f'{dept}_avg_wait'] = dept_waits['duration'].mean()
                
        return metrics

class HospitalSimulation:
    """
    Main simulation controller for hospital operations
    """
    def __init__(self, config_file):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # Initialize simulation environment
        self.env = simpy.Environment()
        
        # Set random seed for reproducibility
        np.random.seed(self.config['simulation']['seed'])
        random.seed(self.config['simulation']['seed'])
        
        # Create departments
        self.departments = {}
        for dept_name, dept_config in self.config['departments'].items():
            dept = Department(self.env, dept_name, dept_config)
            dept.env.simulation = self  # Give department access to the simulation
            self.departments[dept_name] = dept
            
        # Tracking variables
        self.patients = []
        self.completed_patients = []
        self.monitor = PerformanceMonitor()
        self.patient_counter = 0
        
    def generate_patient(self, rate=0.1):
        """
        Generate patients arriving at the hospital
        Returns a generator for the simulation environment
        """
        # Possible patient conditions
        conditions = ["cardiac", "trauma", "respiratory", "general"]
        condition_weights = [0.2, 0.3, 0.25, 0.25]  # Probability distribution
        
        while True:
            # Generate inter-arrival time (exponential distribution)
            interarrival = np.random.exponential(1/rate)
            yield self.env.timeout(interarrival)
            
            # Create new patient
            self.patient_counter += 1
            condition = np.random.choice(conditions, p=condition_weights)
            patient = Patient(self.patient_counter, condition)
            patient.arrival_time = self.env.now
            self.patients.append(patient)
            
            # Start patient journey - all patients start at ER
            self.env.process(self.patient_journey(patient, "ER"))
            
    def patient_journey(self, patient, initial_department):
        """
        Model the complete journey of a patient through the hospital
        """
        current_dept = initial_department
        
        while True:
            # Process treatment in current department
            dept = self.departments[current_dept]
            yield self.env.process(dept.treat_patient(patient))
            
            # Evaluate if transfer is needed
            next_dept, transfer_prob = dept.evaluate_transfer(patient)
            
            # Decide on transfer based on probability
            if next_dept and random.random() < transfer_prob:
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
        
    def run_simulation(self, duration=None, patient_arrival_rate=0.1):
        """Run the simulation for the specified duration"""
        if duration is None:
            duration = self.config['simulation']['duration']
            
        # Start patient generation process
        self.env.process(self.generate_patient(rate=patient_arrival_rate))
        
        # Run simulation
        self.env.run(until=duration)
        
        # Return results
        return self.get_results()
        
    def run_parallel_simulation(self, scenarios=None):
        """
        Run multiple simulation scenarios in parallel
        scenarios: list of (duration, arrival_rate) tuples
        """
        if scenarios is None:
            # Default scenarios: different arrival rates
            scenarios = [
                (self.config['simulation']['duration'], 0.05),  # Low arrival rate
                (self.config['simulation']['duration'], 0.1),   # Medium arrival rate
                (self.config['simulation']['duration'], 0.2)    # High arrival rate
            ]
            
        pool = mp.Pool(processes=min(mp.cpu_count(), len(scenarios)))
        results = pool.starmap(self.run_scenario, scenarios)
        pool.close()
        pool.join()
        
        return results
        
    def run_scenario(self, duration, arrival_rate):
        """Helper method for parallel execution"""
        # Create a new simulation instance with the same config
        sim = HospitalSimulation(self.config)
        return sim.run_simulation(duration, arrival_rate)
        
    def get_results(self):
        """Collect and format simulation results"""
        # Department statistics
        dept_stats = [dept.get_statistics() for dept in self.departments.values()]
        
        # Patient statistics
        patient_stats = []
        for p in self.completed_patients:
            patient_stats.append({
                'id': p.id,
                'condition': p.condition,
                'arrival_time': p.arrival_time,
                'treatment_time': p.get_total_treatment_time(),
                'wait_time': p.get_total_wait_time(),
                'num_transfers': len(p.transfers),
                'num_treatments': len(p.treatment_history)
            })
            
        # Overall metrics
        overall_metrics = self.monitor.calculate_metrics()
        
        return {
            'department_stats': dept_stats,
            'patient_stats': patient_stats,
            'overall_metrics': overall_metrics,
            'completed_patients': len(self.completed_patients),
            'total_patients': len(self.patients)
        }
        
    def set_departments(self, num_departments):
        """
        Helper method to adjust the number of departments for scalability testing
        """
        # This is a simplified implementation for the lab
        # In a real scenario, you'd create the appropriate departments
        dept_names = list(self.config['departments'].keys())
        selected_depts = dept_names[:min(num_departments, len(dept_names))]
        
        # Keep only the selected departments
        self.departments = {name: self.departments[name] for name in selected_depts}
        
    def run_and_analyze(self):
        """Run simulation and return key metrics for optimization"""
        results = self.run_simulation()
        
        # Extract relevant metrics for optimization
        metrics = {
            'avg_wait_time': results['overall_metrics']['avg_wait_time'],
            'resource_utilization': np.mean([
                dept['avg_bed_utilization'] + dept['avg_staff_utilization'] 
                for dept in results['department_stats']
            ]) / 2,  # Average of bed and staff utilization
            'throughput': results['completed_patients'] / self.config['simulation']['duration']
        }
        
        return metrics