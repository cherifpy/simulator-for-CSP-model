import logging
import simpy
import numpy as np

logger = logging.getLogger(__name__)

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw

class ComputeNode:
    def __init__(self, env, node_id, master, bandwidth,compute_capacity=1, energy_consumption=1, storage_capacity=float('inf')):
        self.env = env
        self.node_id = node_id
        self.queue = simpy.Store(env)
        self.master = master
        self.bandwidth = bandwidth
        self.bandwidth_lock = simpy.Resource(env, capacity=bandwidth)
        self.list_of_datasets = []
        self.free_node = True
        self.node_ready_event = None
        self.running_task = []
        self.running_transfer = False
        self.running_job = []
        self.datasets = []
        
        
        self.compute_capacity:float = compute_capacity if master._config['homogeneous'] == False else 1
        self.energy_consumption:float = energy_consumption
        
        self.link_occuped = False
        self.new_task = None
        
            
    def processTasks(self):
        
        while True:
            self.new_task = yield self.queue.get()
            #logger.debug("[%s] Compute-%s: Got new task of job %s: duration: %s", self.env.now, self.node_id, self.new_task.job_id, self.new_task.duration)
            if not self.new_task: continue
            
            if not self.checkForJobs(self.new_task.job_id) and self.new_task != None:
                yield self.env.timeout(0.01)
                yield self.queue.put(self.new_task)

            elif self.new_task != None:
                self.free_node = False

                self.running_task.append(self.new_task)

                if not self.new_task.job_id in self.datasets:
                    yield self.new_task.dataset_ready_event
                    self.datasets.append(self.new_task.job_id)

                start_time = self.env.now
                self.new_task.status = "Started"
                self.new_task.node= self.node_id
                self.new_task.starting_time = start_time
                self.master.tracker.log_task_start(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time)

                task_duration = self.new_task.duration*self.compute_capacity
                yield self.env.timeout(task_duration)  # actual processing
                end_time = self.env.now

                self.master.tracker.log_task_end(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time, end_time)

                self.free_node = True
                self.running_task.pop(0)

    def checkForJobs(self, job_id):
        if job_id not in self.datasets:
            return False
        else:
            return True

    


