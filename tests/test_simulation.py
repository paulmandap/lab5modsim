import pytest # type: ignore
import json
import os
import sys
import numpy as np
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import HospitalSimulation, PerformanceMonitor
from src.patient import Patient
from src.department import Department

@pytest.fixture
def test_config():
    """Create a test configuration file"""
    config = {
        "departments": {
            "ER": {
                "beds": 5,
                "staff": 3,
                "treatment_times": {
                    "cardiac": 30,
                    "trauma": 45,
                    "respiratory": 40,
                    "general": 60
                }
            },
            "ICU": {
                "beds": 3,
                "staff": 4,
                "treatment_times": {
                    "cardiac": 120,
                    "trauma": 180,
                    "respiratory": 150,
                    "general": 90
                }
            }
        },
        "simulation": {
            "duration": 480,  # Shorter duration for tests
            "seed": 42
        }
    }
    
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
        json.dump(config, f)
        config_path = f.name
        
    yield config_path
    
    # Cleanup
    os.unlink(config_path)

def test_patient_creation():
    """Test patient object creation and methods"""
    patient = Patient(1, "cardiac")
    assert patient.id == 1
    assert patient.condition == "cardiac"
    assert patient.status == "waiting"
    
    # Test adding treatment
    patient.add_treatment("ER", 10, 50)
    assert len(patient.treatment_history) == 1
    assert patient.treatment_history[0]['duration'] == 40
    
    # Test adding transfer
    patient.add_transfer("ER", "ICU", 60)
    assert len(patient.transfers) == 1
    assert patient.transfers[0]['from'] == "ER"

def test_patient_flow(test_config):
    """Test basic patient flow through departments"""
    sim = HospitalSimulation(test_config)
    results = sim.run_simulation(duration=120, patient_arrival_rate=0.2)
    
    # Check that some patients completed treatment
    assert len(sim.completed_patients) > 0
    
    # Check that patients went through departments
    for dept_name, dept in sim.departments.items():
        assert dept.patients_treated >= 0

def test_resource_constraints(test_config):
    """Test department resource management"""
    sim = HospitalSimulation(test_config)
    sim.run_simulation(duration=120, patient_arrival_rate=0.3)
    
    # Check that resources weren't over-allocated
    for dept_name, dept in sim.departments.items():
        assert dept.beds.count <= dept.beds.capacity
        assert dept.staff.count <= dept.staff.capacity

def test_performance_monitor():
    """Test the performance monitoring functionality"""
    monitor = PerformanceMonitor()
    
    # Create sample patient
    patient = Patient(1, "trauma")
    
    # Record metrics
    monitor.record_wait_time(patient, 15, "ER")
    monitor.record_occupancy("ER", 10, 0.75)
    monitor.record_transfer(patient, "ER", "ICU", 20)
    
    # Calculate metrics
    metrics = monitor.calculate_metrics()
    
    # Verify metrics
    assert metrics['avg_wait_time'] == 15
    assert metrics['max_wait_time'] == 15
    assert metrics['transfer_rate'] == 1.0  # 1 transfer per patient

def test_parallel_simulation(test_config):
    """Test parallel simulation capabilities"""
    sim = HospitalSimulation(test_config)
    
    # Define a small set of scenarios for testing
    scenarios = [
        (60, 0.1),  # Short simulation, medium arrival rate
        (60, 0.2)   # Short simulation, high arrival rate
    ]
    
    # Run parallel simulations
    results = sim.run_parallel_simulation(scenarios)
    
    # Check results from each scenario
    assert len(results) == 2
    for result in results:
        assert 'department_stats' in result
        assert 'overall_metrics' in result
        
    # Check the high arrival rate led to more patients
    assert results[1]['total_patients'] >= results[0]['total_patients']