import math
import random
import simpy
import numpy as np
import logging
from classes.job import Task, Replica, Job
import copy
from compute_node import ComputeNode

from utils.modelCSP import onLineSchedulingUsingCSP,schedulingUsingJavaCSP


logger = logging.getLogger(__name__)

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw



class SchedulingUsingCSPOnline:
    
    """Master node handles job submissions."""
    def __init__(self, env, compute_nodes, tracker, config, overlap=False):
        self.env = env
        self.queue = simpy.Store(env)
        self.compute_nodes:list[ComputeNode] = compute_nodes
        self.tracker = tracker
        self._config = config
        self.all_jobs = {}
        self.running_jobs:list[Job] = []
        self.waiting_jobs = []
        self.finished_jobs = 0
        self.replicas_stats = {}
        self.replicas_placements = {}
        self.nb_nodes = len(compute_nodes)
        
        self.overlap = self._config['overlap'] if 'overlap' in self._config else overlap
        self.replicas_locations = {}
        self.threshold = config['threshold']
        self.dataset_sizes = []
        self.jobs = []

        logging.debug(f"Master node with {self.nb_nodes} compute nodes started")

    def receiveJobs(self,):
        while True:
            logger.debug("[%s] Master: Waiting for new job", self.env.now)
            new_job = yield self.queue.get()
            new_job.arriving_time = self.env.now

            logger.debug("[%s] Master: Job %s Arrived.", self.env.now, new_job.job_id)

            self.jobs.append(new_job)
            self.waiting_jobs.append(new_job)
            self.tracker.register_job(new_job.job_id, self.env.now)

            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.jobs) == self._config['total_nb_jobs']:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break
                
    def scheduling(self):

        scheduling_start_time = 0
        
        self.ongoing_transfers = {}
        self.ongoing_works = {}
        
        dataset_events = {}
        transfers, works = None, None        
        block = False

        for node_id in range(len(self.compute_nodes)):
            self.ongoing_transfers[f'node_{node_id}'] = None
            self.ongoing_works[f'node_{node_id}'] = None
        
        t_now = self.env.now
        
        while True:
            yield self.env.timeout(0.1)

            """
                Scheduling loop.
                Here the condistion is a leat we have one waiting job on the master node.
                This condition can be changed to have a periodic scheduling or any other condition.
            """    

            if len(self.waiting_jobs) >= 1:

                t_now = self.env.now

                nodes_free_time = self.nodesFreeTime(self.ongoing_transfers, self.ongoing_works, scheduling_start_time)
                replicas_locations = self.replicas_locations
                
                """ 
                    here i select the jobs to reschedule 
                    In this case we only schedule a job at time + the actual running jobs 
                """
                jobs_to_reschedule =[self.waiting_jobs[0]] + self.getRunningJobs() 
                
                if len(jobs_to_reschedule) > 0:
                    
                    logger.debug("[%s] Master: Start looking for a solution. at time %s", self.env.now,self.env.now)
                    transfers_, works_ = schedulingUsingJavaCSP(self, jobs_to_reschedule, replicas_locations, nodes_free_time)

                else:
                    transfers_, works_ = {}, {}
                
                if len(transfers_.keys()) > 0 and len(works_.keys()) > 0:
                    
                    transfers, works = copy.copy(transfers_), copy.copy(works_)
                    scheduling_start_time = int(self.env.now) + 1
                    
                    """
                        # remove the scheduled job from the waiting list
                    """
                    for i in range(1): 
                        if len(self.waiting_jobs) > 0:
                            self.waiting_jobs.pop(0)

                    self.ongoing_transfers = {}
                    self.ongoing_works = {}                
                    
                    for node_id in range(len(self.compute_nodes)):
                        self.ongoing_transfers[f'node_{node_id}'] = None
                        self.ongoing_works[f'node_{node_id}'] = None
                
                else:
                    print('no solution found at time ', self.env.now)


            if not transfers and not works:
                continue
            
            for node_id in range(len(self.compute_nodes)):    
            
                if len(transfers[f'node_{node_id}']) > 0:
                    if self.ongoing_transfers[f'node_{node_id}'] is None:
                        
                        (job_id, _, t_start, t_end, duration) = transfers[f'node_{node_id}'][0]
                        
                        if scheduling_start_time + t_start <= self.env.now:
                        
                            (job_id, _, t_start, t_end, duration) = transfers[f'node_{node_id}'].pop(0)
                            transfer_time = transferCost(self, self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth, self._config)
                        
                            t_s = int(self.env.now)+1
                        
                            self.ongoing_transfers[f'node_{node_id}'] = (job_id, node_id, t_s, t_s + transfer_time, transfer_time)
                        
                            dataset_events[(node_id, job_id)] = self.env.event()
                        
                            self.startTransfer(self.compute_nodes[node_id], self.jobs[job_id],dataset_events[(node_id, job_id)])
                        
                            logger.debug("[%s] Master: Transfer of job %s to node %s started.", self.env.now, job_id, node_id)

                "Check if any transfer is finished"
                if self.ongoing_transfers[f'node_{node_id}'] is not None:
                    job_id, _, t_start, t_end, duration =  self.ongoing_transfers[f'node_{node_id}']
                    if self.env.now >=  t_start + transferCost(self,self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth):
                        self.ongoing_transfers[f'node_{node_id}'] = None
                            
                if len(works[f'node_{node_id}']) > 0:
                    if self.ongoing_works[f'node_{node_id}'] is None:
                        
                        (job_id, _,k, t_start, t_end, duration) = works[f'node_{node_id}'][0]
                        
                        if scheduling_start_time + t_start <= self.env.now and (node_id, job_id) in dataset_events.keys():
                            (job_id, _,k, t_start, t_end, duration) = works[f'node_{node_id}'].pop(0)
                            
                            job = self.jobs[job_id]
                            
                            compute_node = self.compute_nodes[node_id]
                        
                            not_executed_tasks = [task for task in job.tasks if task.status == "NotStarted"]
                            
                            task = None if len(not_executed_tasks) == 0 else not_executed_tasks[0]

                            if task:
                                task.dataset_ready_event = dataset_events[(node_id, job_id)]
                                job.nb_remaining_tasks -= 1
                                self.ongoing_works[f'node_{node_id}'] = (job_id, node_id, task.task_id, t_start, t_end, duration)
                                task.node = compute_node.node_id
                                self.replicas_stats[(job.job_id, task.node)].nb_tasks +=1
                                self.replicas_stats[(job.job_id, task.node)].task_execution_time += task.duration*compute_node.compute_capacity
                                task.status = "Scheduled"
                                
                                yield compute_node.queue.put(task)

                                logger.debug("[%s] Master : Sent task %s from job %s to node %s", self.env.now, task.task_id, job.job_id, task.node)

                if self.ongoing_works[f'node_{node_id}'] is not None:
                    job_id, _, k, t_start, t_end, d =  self.ongoing_works[f'node_{node_id}']
                    task = self.jobs[job_id].tasks[k]
                    if task.status == "Finished":
                        self.ongoing_works[f'node_{node_id}'] = None

            """
                Condition to exit the scheduling loop.
            """
            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.jobs) == self._config['total_nb_jobs']:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break
    
    def checkOnJobs(self,):

        while True:
            yield self.env.timeout(0.1)
            for job in self.jobs:
                if job.status != "Finished":
                    finished_tasks = [task for task in job.tasks if task.status == "Started"]
                    for task in finished_tasks:
                        if task.starting_time + task.duration*self.compute_nodes[task.node].compute_capacity <= self.env.now:
                            task.status = "Finished"
                            task.finishing_time = self.env.now
                            job.task_execution_time = task.duration
                            logger.debug("[%s] Master: Task %s of job %s finished on node %s.", self.env.now, task.task_id, job.job_id, task.node)
                               

            for job in self.jobs:
               finished_tasks = [task for task in job.tasks if task.status == "Finished"]
               if len(finished_tasks) == len(job.tasks) and job.status!="Finished":
                   self.finished_jobs += 1
                   job.finish_time = np.max([task.finishing_time for task in job.tasks])
                   job.status = "Finished"
                   self.tracker.log_end_job(job.job_id,len(job.tasks),job.dataset_size,job.arriving_time,job.starting_time, job.finish_time,job.transfer_time, job.tasks[0].duration, job.nb_replicas, job.first_optimal_replica_number,job.nb_first_replicas_sended)
                   logger.debug("[%s] Master: Job %s finished.", self.env.now, job.job_id)

            
            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.jobs) == self._config['total_nb_jobs']:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break



    def startTransfer(self,compute_node, job, event):
        self.updateRunningJobs(job)

        job.replicas_nodes.append(compute_node.node_id)
        job.transfer_time =  transferCost(self,job.dataset_size,compute_node.bandwidth) # new_job.dataset_size / 
        replica_inst =  Replica(job.job_id, node_id=compute_node.node_id, data_size=job.dataset_size, 
                                transfer_time=job.transfer_time,transfer_start_time=self.env.now)
        self.replicas_stats[(job.job_id, compute_node.node_id)] = replica_inst
        job.replicas.append(replica_inst)
        
        dataset_ready_event = event

        self.env.process(self.transferData(job.job_id, job.dataset_size, compute_node,dataset_ready_event))
        
        job.nb_replicas +=1
        job.node_referent = compute_node.node_id

    def getRunningJobs(self):
        to_reschedule = []
        for job in self.jobs:
            not_eexecuted_tasks = [task for task in job.tasks if task.status == "NotStarted"]
            if job.status != "Finished" and len(not_eexecuted_tasks) > 0:
                to_add = True
                for w_job in self.waiting_jobs:
                    if job.job_id == w_job.job_id:
                        to_add = False
                        break
                if to_add: to_reschedule.append(job)
        return to_reschedule

    def updateWaitingList(self, job_id):
        to_delete = None
        for i, job in enumerate(self.waiting_jobs):
            if job.job_id == job_id:
                to_delete = i
                break
        if to_delete: self.waiting_jobs.pop(i)

    def isNoJobRunning(self,job_id, transfers, works):
        if not transfers or not works:
            return True
        for node_id in range(len(self.compute_nodes)):
            if self.ongoing_transfers[f'node_{node_id}'] is not None or self.ongoing_works[f'node_{node_id}'] is not None \
                or (f'node_{node_id}' in transfers.keys() and len(transfers[f'node_{node_id}']) > 0) and (f'node_{node_id}' in works.keys() and len(works[f'node_{node_id}']) > 0):
                return False

        if len([job for job in self.jobs if job.status != "Finished" and job.job_id < job_id]) > 0:
            return False

        return True
    
    def updateReplicasLocation(self, job_id, node_id):
        if job_id in self.self.replicas_locations.keys() and node_id not in self.self.replicas_locations[job_id]:
            self.replicas_locations[job_id].append(node_id)
        else:
            self.replicas_locations[job_id] = [node_id]

    def updateRunningJobs(self, job):
        if job.job_id not in [j.job_id for j in self.running_jobs]:
            self.running_jobs.append(job)

    def nodesFreeTime(self, ongoing_transfers, ongoing_works, scheduling_start_time):
        nodes_free_time = {}
        for node_id in range(len(self.compute_nodes)):
            if f'node_{node_id}' in ongoing_transfers.keys() and ongoing_transfers[f'node_{node_id}'] is not None:
                _, node_id, _, t_end, duration =  ongoing_transfers[f'node_{node_id}']    
                nodes_free_time[node_id] = int(t_end - self.env.now)
                
            
            if f'node_{node_id}' in ongoing_works.keys() and ongoing_works[f'node_{node_id}'] is not None:
                
                job_id, node_id, k, t_start, t_end, duration =  ongoing_works[f'node_{node_id}']
                
                task = self.jobs[job_id].tasks[k]
                if task.status == "Started":
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = int(t_end - self.env.now)
                if task.status == "Finished":
                    nodes_free_time[node_id] = 0
                else:
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = execution_time

            if ongoing_works[f'node_{node_id}'] is None or ongoing_transfers[f'node_{node_id}'] is None:
                nodes_free_time[node_id] = 0
        return nodes_free_time

    def transferData(self, job_id, dataset_size, compute_node, dataset_ready_event,task_id= -1, send_task = False):

        if job_id in self.replicas_locations.keys() and compute_node.node_id in self.replicas_locations[job_id]:
            dataset_ready_event.succeed()
            return

        elif job_id not in self.replicas_locations.keys():
            self.replicas_locations[job_id] = []

        self.replicas_locations[job_id].append(compute_node.node_id)

        transfer_time = dataset_size / compute_node.bandwidth
        
        if self.jobs[job_id].starting_time is None: self.jobs[job_id].starting_time = self.env.now

        with compute_node.bandwidth_lock.request() as node_req:
            
            yield node_req  
            
            logger.debug("[%s] Compute-%s: Got new dataset transfert of job %s: duration: %s, dataset_size: %s", self.env.now, compute_node.node_id, job_id, transfer_time, dataset_size)
            yield self.env.timeout(transfer_time)
            end_time = self.env.now            
            self.compute_nodes[compute_node.node_id].datasets.append(job_id)
            
            dataset_ready_event.succeed()

            self.tracker.log_transfer(job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id)

            
