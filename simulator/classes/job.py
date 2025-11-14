class Task:
    """Task class representing a unit of work within a job."""
    def __init__(self, job_id,  task_id, duration, dataset_size):
        self.job_id = job_id
        self.task_id = task_id
        self.duration = duration
        self.dataset_size = dataset_size
        self.dataset_ready_event = None
        self.status = "NotStarted"
        self.node = None
        self.starting_time = None
        self.finishing_time = None
        self.validated = False
        

class Replica:
    def __init__(self, job_id, node_id, data_size, transfer_time,transfer_start_time, time_to_start=None):
        self.job_id = job_id
        self.node_id = node_id
        self.nb_tasks = 0
        self.transfer_time = transfer_time
        self.data_size = data_size
        self.task_execution_time = 0
        self.transfer_start_time = transfer_start_time
        self.time_to_start = time_to_start
        self.nb_task_to_sitisfy = 0

class Job:
    """Job class representing a collection of tasks sharing the same dataset."""
    def __init__(self, job_id, tasks_duration, nb_tasks, dataset_size):
        self.job_id = job_id
        self.tasks = [Task(job_id=job_id, task_id=i, duration=tasks_duration, dataset_size=dataset_size)
                      for i in range(nb_tasks)]
        self.dataset_size = dataset_size
        self.nb_remaining_tasks = nb_tasks
        
        self.status = "NotStarted" #Notstarted Running Finished

        self.nb_replicas = 0
        self.replicas:list[Replica] = []
        
        self.arriving_time = None
        self.starting_time = None
        self.finish_time = None

        self.transfer_time = None
        self.ids_executed_tasks = []
        self.replicas_nodes = []
        self.events_list = []
        
        self.first_optimal_replica_number = None
        self.task_execution_time = None
        self.node_referent = None
        self.last_event = None
        
        self.nb_tasks = len(self.tasks)
        self.final_replicas_number  = None 
        self.optimal_replicas_number = None
        self.nb_first_replicas_sended = None
        self.idle_replicas = []
        self.first_task_starting_time = None
        self.params = None


    def getNoTExecutedTasks(self):
        not_executed_tasks = []
        for task in self.tasks:
            if task.status=="NotStarted":
                not_executed_tasks.append(task)
        
        return not_executed_tasks

        #return [task for task in self.tasks if task.status=="NotStarted"]

    def getRemainingTasks(self):
        not_executed_tasks = []
        for task in self.tasks:
            if task.status in ["NotStated", "Scheduled"]:
                not_executed_tasks.append(task)
        
        return not_executed_tasks

        #return [task for task in self.tasks if task.status=="NotStarted"]


    def getScheduledTasks(self):
        not_executed_tasks = []
        for task in self.tasks:
            if task.status=="Scheduled":
                not_executed_tasks.append(task)
        
        return not_executed_tasks

        #return [task for task in self.tasks if task.status=="NotStarted"]
