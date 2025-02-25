import simpy
import numpy as np
from collections import defaultdict

class Department:
    """
    Represents a hospital department with resources and treatment capabilities
    """
    def __init__(self, env, name, config):
        self.env = env
        self.name = name
        self.config = config
        
        # Create resources
        self.beds = simpy.Resource(env, capacity=config['beds'])
        self.staff = simpy.Resource(env, capacity=config['staff'])
        
        # Treatment times by condition
        self.treatment_times = config['treatment_times']
        
        # Monitoring metrics
        self.patients_treated = 0
        self.patients_transferred = 0
        self.wait_times = []
        self.occupancy_history = []
        self.queue_length_history = []
        
        # Record occupancy over time
        self.env.process(self.monitor_occupancy())
        
    def monitor_occupancy(self):
        """Records bed and staff utilization at regular intervals"""
        while True:
            self.occupancy_history.append({
                'time': self.env.now,
                'bed_utilization': self.beds.count / self.beds.capacity,
                'staff_utilization': self.staff.count / self.staff.capacity,
                'bed_queue': len(self.beds.queue),
                'staff_queue': len(self.staff.queue)
            })
            self.queue_length_history.append({
                'time': self.env.now,
                'bed_queue': len(self.beds.queue),
                'staff_queue': len(self.staff.queue)
            })
            yield self.env.timeout(10)  # Record every 10 time units
    
    def get_treatment_time(self, condition):
        """
        Determine treatment time based on patient condition
        with some random variation
        """
        base_time = self.treatment_times.get(condition, self.treatment_times["general"])
        # Add 20% random variation
        return max(1, np.random.normal(base_time, base_time * 0.2))
        
    def treat_patient(self, patient):
        """
        Process patient treatment, acquiring necessary resources
        """
        arrival = self.env.now
        patient.current_department = self.name
        
        # Request bed
        with self.beds.request() as bed_request:
            yield bed_request
            bed_acquisition_time = self.env.now
            
            # Record wait time for bed
            bed_wait_time = bed_acquisition_time - arrival
            self.wait_times.append({
                'patient_id': patient.id,
                'resource': 'bed',
                'wait_time': bed_wait_time
            })
            
            # Request staff
            with self.staff.request() as staff_request:
                yield staff_request
                staff_acquisition_time = self.env.now
                
                # Record wait time for staff
                staff_wait_time = staff_acquisition_time - bed_acquisition_time
                self.wait_times.append({
                    'patient_id': patient.id,
                    'resource': 'staff',
                    'wait_time': staff_wait_time
                })
                
                # Start treatment
                treatment_start = self.env.now
                patient.status = "in_treatment"
                
                # Determine treatment duration
                treatment_duration = self.get_treatment_time(patient.condition)
                
                # Simulate treatment time
                yield self.env.timeout(treatment_duration)
                
                # Record treatment
                treatment_end = self.env.now
                patient.add_treatment(self.name, treatment_start, treatment_end)
                self.patients_treated += 1
        
        # After treatment, patient is ready for discharge or transfer
        patient.status = "post_treatment"
        
    def evaluate_transfer(self, patient):
        """
        Determine if patient needs transfer to another department
        based on their condition and treatment history.
        """
        # First check if the transfer department exists in the simulation
        # to avoid KeyError exceptions

        if patient.condition == "cardiac" and self.name != "ICU":
            # Check if ICU exists before attempting transfer
            if "ICU" in self.env.simulation.departments:
                return "ICU", 0.7  # 70% transfer probability to ICU
            else:
                return "General", 0.7  # Fallback to General if ICU doesn't exist

        elif patient.condition == "trauma" and self.name == "ER":
            # Check if General exists
            if "General" in self.env.simulation.departments:
                return "General", 0.8  # 80% transfer probability to General
            else:
                return None, 0  # No transfer if destination doesn't exist

        elif len(patient.treatment_history) >= 2 and self.name != "General":
            # Check if General exists
            if "General" in self.env.simulation.departments:
                return "General", 0.5  # After 2 treatments, 50% move to General
            else:
                return None, 0  # No transfer if destination doesn't exist

        return None, 0  # No transfer needed

            
    def get_statistics(self):
        """Return department performance statistics"""
        if not self.wait_times:
            return {
                'department': self.name,
                'patients_treated': self.patients_treated,
                'patients_transferred': self.patients_transferred,
                'avg_wait_time': 0,
                'max_wait_time': 0,
                'avg_bed_utilization': 0,
                'avg_staff_utilization': 0
            }
            
        avg_wait = np.mean([w['wait_time'] for w in self.wait_times])
        max_wait = max([w['wait_time'] for w in self.wait_times]) if self.wait_times else 0
        
        if not self.occupancy_history:
            return {
                'department': self.name,
                'patients_treated': self.patients_treated,
                'patients_transferred': self.patients_transferred,
                'avg_wait_time': avg_wait,
                'max_wait_time': max_wait,
                'avg_bed_utilization': 0,
                'avg_staff_utilization': 0
            }
            
        avg_bed_util = np.mean([o['bed_utilization'] for o in self.occupancy_history])
        avg_staff_util = np.mean([o['staff_utilization'] for o in self.occupancy_history])
        
        return {
            'department': self.name,
            'patients_treated': self.patients_treated,
            'patients_transferred': self.patients_transferred,
            'avg_wait_time': avg_wait,
            'max_wait_time': max_wait,
            'avg_bed_utilization': avg_bed_util,
            'avg_staff_utilization': avg_staff_util
        }