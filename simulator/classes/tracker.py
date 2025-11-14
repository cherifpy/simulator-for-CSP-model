import logging
logger = logging.getLogger(__name__)

class Tracker:
    """Monitor and log what's happening"""
    def __init__(self, env):
        self.env = env
        self.ongoing_tasks = {}  # Monitoring
        self.ongoing_jobs = {}  # Monitoring
        self.events_history = []  # For Gantt chart

        self.total_nb_transfers = 0
        self.total_nb_transferred_bytes = 0
        self.total_nb_tasks_processed = 0
        self.total_tasks_duration = 0
        self.total_wall_time = 0
        self.tasks_duration_per_job = {}

        self.stats_on_replicas = []
        self.stats_on_tasks = []
        self.stats_on_jobs = []
        self.threshold_history = []

    def register_job(self, job_id, start_time):
        if job_id not in self.tasks_duration_per_job:
            self.tasks_duration_per_job[job_id] = {'start_time': start_time, 'end_time': 0}

    def log_transfer(self, job_id, node_id, start_time, end_time, dataset_size, task_id=-1):
        self.events_history.append({
            'job_id': job_id,
            'node_id': node_id,
            'type': 'transfer',
            'start': start_time,
            'end': end_time,
            'transferred_bytes': dataset_size
        })

        logger.debug("[%s] Transfer for job %s to node %s took %s",
                     self.env.now, job_id, node_id, end_time-start_time)
        self.stats_on_tasks.append({
            "job":job_id,
            "task":-1,
            "node":node_id,
            "start_time":start_time,
            "finishing_time":end_time,
            "execution_time":end_time-start_time,
            "id_dataset":job_id,
            "transfert_time":end_time-start_time,
        })
        self.total_nb_transfers += 1
        self.total_nb_transferred_bytes += dataset_size


    def log_task_start(self, job_id, task_id, node_id, start_time):
        self.events_history.append({
            'job_id': job_id,
            'task_id': task_id,
            'node_id': node_id,
            'type': 'processing',
            'start': start_time})

        # Add the current task to the ongoing tasks
        self.ongoing_tasks[str(job_id)+"-"+str(task_id)] = {'node_id': node_id,'start_time': start_time}
        
        logger.debug("[%s] Task %s of Job %s started on Node-%s.", self.env.now, task_id, job_id, node_id)

    def log_task_end(self, job_id, task_id, node_id, start_time, end_time):
        self.events_history.append({
            'job_id': job_id,
            'task_id': task_id,
            'node_id': node_id,
            'type': 'processing',
            'start': start_time,
            'end': end_time})

        self.stats_on_tasks.append({
            "job":job_id,
            "task":task_id,
            "node":node_id,
            "start_time":start_time,
            "finishing_time":end_time,
            "execution_time":end_time-start_time,
            "id_dataset":job_id,
            "transfert_time":0,
        })

        # Remove the current task from the ongoing tasks
        if str(job_id)+"-"+str(task_id) in self.ongoing_tasks:
            del self.ongoing_tasks[str(job_id)+"-"+str(task_id)]

        logger.debug("[%s] Task %s of Job %s completed on Node-%s.", self.env.now, task_id, job_id, node_id)
        self.total_nb_tasks_processed += 1
        self.total_tasks_duration += (end_time - start_time)
        self.total_wall_time = max(end_time, self.total_wall_time)
        self.tasks_duration_per_job[job_id]['end_time'] = max(end_time, self.tasks_duration_per_job[job_id]['end_time'])

    def log_end_job(self, job_id, nb_task, dataset_size, arriving_time, start_time, end_time, transfert_time, task_execution_time,
                    nb_replicas,optimal_replicas_number=0,nb_first_replicas_sended=0):
        self.stats_on_jobs.append({
            "job_id":job_id,
            "nb_tasks":nb_task,
            "task_execution_time":task_execution_time,
            "arriving_time":arriving_time,
            "starting_time":start_time,
            "finishing_time":end_time,
            "dataset size":dataset_size,
            "transfert time":transfert_time,
            "nb replicas":nb_replicas,
            "optimal_replicas_number":optimal_replicas_number,
            "nb_first_replicas_sended":nb_first_replicas_sended
        })

        logger.debug("[%s] Job %s finished.", self.env.now, job_id)
        #print(f"Job {job_id} stats: nb_tasks={nb_task}, dataset_size={dataset_size}, arriving_time={arriving_time}, starting_time={start_time}, finishing_time={end_time}, transfert_time={transfert_time}, task_execution_time={task_execution_time}, nb_replicas={nb_replicas}")
    def log_threshold(self, theta):
        self.threshold_history.append({"time": self.env.now, "theta": theta})
        
