class Patient:
    """
    Represents a patient in the hospital system
    Tracks patient attributes and treatment history
    """
    def __init__(self, patient_id, condition):
        self.id = patient_id
        self.condition = condition
        self.arrival_time = None
        self.treatment_history = []
        self.transfers = []
        self.current_department = None
        self.status = "waiting"  # waiting, in_treatment, discharged

    def add_treatment(self, department, start_time, end_time):
        """Records treatment episodes"""
        self.treatment_history.append({
            'department': department,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })

    def add_transfer(self, from_dept, to_dept, transfer_time):
        """Records patient transfers between departments"""
        self.transfers.append({
            'from': from_dept,
            'to': to_dept,
            'time': transfer_time
        })
        
    def get_total_treatment_time(self):
        """Calculate the total time spent in treatment"""
        return sum(episode['duration'] for episode in self.treatment_history)
    
    def get_total_wait_time(self):
        """Calculate total wait time (excluding treatment time)"""
        if not self.treatment_history:
            return 0
            
        total_time = 0
        last_end = self.arrival_time
        
        for episode in self.treatment_history:
            total_time += episode['start_time'] - last_end
            last_end = episode['end_time']
            
        return total_time