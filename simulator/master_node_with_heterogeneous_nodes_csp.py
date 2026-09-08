import math
import random
import simpy
import numpy as np
import logging
from classes.job import Task, Replica, Job
import copy
from compute_node import ComputeNode

from utils.modelCSP import onLineSchedulingUsingCSP,schedulingUsingJavaCSP,startMinizincModel


logger = logging.getLogger(__name__)

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw



class SchedulingUsingCSPOnline:

    """Master node: receives job submissions and drives CSP-based scheduling over a set of heterogeneous compute nodes."""
    # Dedicated Java entry point (utils/model/src/main/MainOnline.java) so Online-specific
    # storage handling can evolve independently of Incremental's.
    java_main_class = 'MainOnline'
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

        self.nb_nodes = len(compute_nodes)

        self.overlap = self._config['overlap'] if 'overlap' in self._config else overlap
        self.replicas_locations = {}
        self.threshold = config['threshold']
        self.dataset_sizes = []
        self.jobs = []

        self.last_scheduling_time = 0
        # Per-node plan produced by the last CSP solve: transfers/tasks still queued to start.
        self.transfers, self.works = {}, {}
        # Per-node queue of (job_id, deletion_time): replicas the CSP chose to abandon rather
        # than keep indefinitely, to actually be freed from that node's storage.
        self.deletions = {}
        # Per-node bookkeeping of what is currently executing (at most one transfer and one
        # task tracked per node at any given time).
        self.ongoing_transfers = {}
        self.ongoing_works = {}

        for node_id in range(len(self.compute_nodes)):
            self.ongoing_transfers[f'node_{node_id}'] = None
            self.ongoing_works[f'node_{node_id}'] = None
            self.transfers[f'node_{node_id}'] = []
            self.works[f'node_{node_id}'] = []
            self.deletions[f'node_{node_id}'] = []

        self.dataset_events = {}

        logger.debug("Master node started with %s compute nodes", self.nb_nodes)

    def _allJobsCompleted(self):
        """True once every submitted job has finished and no compute node still has queued work."""
        if self.finished_jobs != self._config['total_nb_jobs']:
            return False
        if self.waiting_jobs or len(self.jobs) != self._config['total_nb_jobs']:
            return False
        return all(len(compute_node.queue.items) == 0 for compute_node in self.compute_nodes)

    def receiveJobs(self,):
        self.ongoing_transfers = {}
        self.ongoing_works = {}

        for node_id in range(len(self.compute_nodes)):
            self.ongoing_transfers[f'node_{node_id}'] = None
            self.ongoing_works[f'node_{node_id}'] = None
            # compute_nodes is still [] at __init__ time, so these per-node keys need to be
            # (re)seeded here once the real node list is known. Online always overwrites every
            # key on each replan so this is a no-op for it, but Incremental only appends -- it
            # needs every node's key to already exist before scheduling()/nodesFreeTimeIncremental
            # index into it.
            self.transfers.setdefault(f'node_{node_id}', [])
            self.works.setdefault(f'node_{node_id}', [])
            self.deletions.setdefault(f'node_{node_id}', [])
        while True:
            logger.debug("[%s] Master: waiting for new job", self.env.now)
            new_job = yield self.queue.get()
            new_job.arriving_time = self.env.now

            logger.debug("[%s] Master: job %s arrived", self.env.now, new_job.job_id)

            self.jobs.append(new_job)
            self.waiting_jobs.append(new_job)
            self.tracker.register_job(new_job.job_id, self.env.now)

            if self._allJobsCompleted():
                break

    def schedulingNewJob(self):

        while True:
            yield self.env.timeout(0.1)

            # Scheduling loop: triggers as soon as at least one job is waiting. This condition
            # could be replaced by a periodic trigger or any other policy.
            if len(self.waiting_jobs) >= 1:

                nodes_free_time = self.nodesFreeTime(self.ongoing_transfers, self.ongoing_works)

                replicas_locations = self.replicas_locations

                # Jobs to (re)schedule: the waiting job(s) plus every already-running job that
                # still has unstarted tasks. The whole placement is recomputed from scratch,
                # nothing is scheduled incrementally on top of the previous solution.
                jobs_to_reschedule = [job for job in self.waiting_jobs] + self.getRunningJobs()

                if len(jobs_to_reschedule) > 0:
                    logger.debug("[%s] Master: looking for a solution for %s job(s)", self.env.now, len(jobs_to_reschedule))
                    if self._config['use_minizinc_model']:
                        transfers_, works_, deletions_ = startMinizincModel(self, jobs_to_reschedule, replicas_locations, nodes_free_time)

                    else:
                        transfers_, works_, deletions_ = schedulingUsingJavaCSP(self, jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)
                else:
                    transfers_, works_, deletions_ = {}, {}, {}


                if len(transfers_.keys()) > 0 and len(works_.keys()) > 0:
                    for node in range(len(self.compute_nodes)):
                        key = "node_" + str(node)
                        self.transfers[key] = []
                        self.works[key] = []
                        self.deletions[key] = []
                        if key in transfers_.keys() and len(transfers_[key]) > 0:
                            ongoing = self.ongoing_transfers.get(key)
                            for transfer in transfers_[key]:
                                transfer_job_id = transfer[0]
                                # The CSP re-emits a transfer for every (node, job) pair it still
                                # wants to keep, even when the data is already resident there (or
                                # already mid-transfer) from an earlier solve -- its own timing for
                                # that phantom entry is a free, cost-free variable with no guarantee
                                # of matching reality. Queuing it again only delays dispatch of the
                                # work waiting on this node's dataset_ready_event until this phantom
                                # transfer is (pointlessly) dequeued -- so skip it and let the
                                # already-fired (or still in-flight) event from the real transfer
                                # keep gating that work instead.
                                already_present = transfer_job_id in self.replicas_locations and node in self.replicas_locations[transfer_job_id]
                                already_in_flight = ongoing is not None and ongoing[0] == transfer_job_id
                                if already_present or already_in_flight:
                                    continue
                                self.transfers[key].append(transfer)

                            if len(works_[key]) > 0:
                                for work in works_[key]:
                                    self.works[key].append(work)

                        if key in deletions_.keys() and len(deletions_[key]) > 0:
                            for deletion in deletions_[key]:
                                self.deletions[key].append(deletion)

                    # The whole batch was just (re)planned: nothing is left waiting.
                    self.waiting_jobs.clear()

                else:
                    logger.warning("[%s] Master: no CSP solution found for %s job(s)", self.env.now, len(jobs_to_reschedule))

            if self._allJobsCompleted():
                break

    def scheduling(self):

        while True:
            yield self.env.timeout(0.1)

            if not self.transfers and not self.works:
                continue

            for node_id in range(len(self.compute_nodes)):

                # Free up this node's transfer slot once its ongoing transfer has completed.
                if self.ongoing_transfers[f'node_{node_id}'] is not None:
                    job_id, _, t_start, t_end, duration = self.ongoing_transfers[f'node_{node_id}']
                    if self.env.now >= t_start + duration:
                        self.ongoing_transfers[f'node_{node_id}'] = None

                if len(self.transfers[f'node_{node_id}']) > 0:
                    if self.ongoing_transfers[f'node_{node_id}'] is None:

                        (job_id, _, t_start, _, duration) = self.transfers[f'node_{node_id}'][0]

                        # Only pop and start once the planned start time is actually reached.
                        # TODO: revisit - the CSP's planned t_start can lag behind env.now by up
                        # to one scheduling tick, worth double-checking the edge case here.
                        if t_start <= self.env.now:

                            (job_id, _, t_start, t_end, duration) = self.transfers[f'node_{node_id}'].pop(0)

                            self.ongoing_transfers[f'node_{node_id}'] = (job_id, node_id, t_start, t_end, duration)

                            self.dataset_events[(node_id, job_id)] = self.env.event()

                            self.startTransfer(self.compute_nodes[node_id], self.jobs[job_id], self.dataset_events[(node_id, job_id)], duration=duration)

                            logger.debug("[%s] Master: transfer of job %s to node %s started", self.env.now, job_id, node_id)

                # Free up this node's work slot once its ongoing task has completed.
                if self.ongoing_works[f'node_{node_id}'] is not None:
                    job_id, _, k, _, _, _ = self.ongoing_works[f'node_{node_id}']
                    task = self.jobs[job_id].tasks[k]
                    if task.status == "Finished":
                        self.ongoing_works[f'node_{node_id}'] = None

                        # The job's data is only needed on this node for as long as one of
                        # its tasks is still queued to run here. No task of this job left in
                        # this node's queue -> the data can be released right away, instead of
                        # waiting on a CSP-decided deletion time that can go stale once the job
                        # drops out of the reconsideration batch (no "NotStarted" tasks left) and
                        # is never revisited by a later solve.
                        still_needed_here = any(w[0] == job_id for w in self.works[f'node_{node_id}'])
                        if not still_needed_here and job_id in self.replicas_locations and node_id in self.replicas_locations[job_id]:
                            compute_node = self.compute_nodes[node_id]
                            self.replicas_locations[job_id].remove(node_id)
                            if job_id in compute_node.datasets:
                                compute_node.datasets.remove(job_id)
                            # Drop any CSP-scheduled deletion still pending for this pair so it
                            # doesn't fire a redundant (and now stale) deletion later.
                            self.deletions[f'node_{node_id}'] = [
                                d for d in self.deletions[f'node_{node_id}'] if d[0] != job_id
                            ]
                            self.tracker.log_deletion(job_id, node_id, self.env.now)
                            logger.debug("[%s] Master: released job %s's data from node %s (no task of this job left there)", self.env.now, job_id, node_id)

                if len(self.works[f'node_{node_id}']) > 0 and self.ongoing_works[f'node_{node_id}'] is None:

                    (job_id, _, k, t_start, _, _) = self.works[f'node_{node_id}'][0]

                    if t_start <= self.env.now and (node_id, job_id) in self.dataset_events.keys():
                        (job_id, _, k, t_start, t_end, duration) = self.works[f'node_{node_id}'].pop(0)

                        job = self.jobs[job_id]

                        compute_node = self.compute_nodes[node_id]

                        not_executed_tasks = [task for task in job.tasks if task.status == "NotStarted"]

                        task = None if len(not_executed_tasks) == 0 else not_executed_tasks[0]

                        if task:
                            task.dataset_ready_event = self.dataset_events[(node_id, job_id)]
                            job.nb_remaining_tasks -= 1

                            self.ongoing_works[f'node_{node_id}'] = (job_id, node_id, task.task_id, t_start, t_start + duration, duration)
                            task.node = compute_node.node_id
                            self.replicas_stats[(job.job_id, task.node)].nb_tasks += 1
                            self.replicas_stats[(job.job_id, task.node)].task_execution_time += task.duration * compute_node.compute_capacity
                            task.status = "Scheduled"

                            yield compute_node.queue.put(task)

                            logger.debug("[%s] Master: sent task %s of job %s to node %s", self.env.now, task.task_id, job.job_id, task.node)

                # Actually free this node's storage for replicas the CSP chose to abandon
                # (rather than keep indefinitely) once their scheduled deletion time is reached.
                while self.deletions[f'node_{node_id}'] and self.deletions[f'node_{node_id}'][0][1] <= self.env.now:
                    job_id, deletion_time = self.deletions[f'node_{node_id}'].pop(0)
                    compute_node = self.compute_nodes[node_id]

                    if job_id in self.replicas_locations and node_id in self.replicas_locations[job_id]:
                        self.replicas_locations[job_id].remove(node_id)
                    if job_id in compute_node.datasets:
                        compute_node.datasets.remove(job_id)

                    self.tracker.log_deletion(job_id, node_id, self.env.now)
                    logger.debug("[%s] Master: deleted job %s's data from node %s", self.env.now, job_id, node_id)

            if self._allJobsCompleted():
                break

    def checkOnJobs(self,):

        while True:
            yield self.env.timeout(0.1)
            for job in self.jobs:
                if job.status != "Finished":
                    started_tasks = [task for task in job.tasks if task.status == "Started"]
                    for task in started_tasks:
                        if task.starting_time + task.duration * self.compute_nodes[task.node].compute_capacity <= self.env.now:
                            task.status = "Finished"
                            task.finishing_time = self.env.now
                            job.task_execution_time = task.duration
                            logger.debug("[%s] Master: task %s of job %s finished on node %s", self.env.now, task.task_id, job.job_id, task.node)

            for job in self.jobs:
                finished_tasks = [task for task in job.tasks if task.status == "Finished"]
                if len(finished_tasks) == len(job.tasks) and job.status != "Finished":
                    self.finished_jobs += 1
                    job.finish_time = np.max([task.finishing_time for task in job.tasks])
                    job.status = "Finished"
                    self.tracker.log_end_job(job.job_id, len(job.tasks), job.dataset_size, job.arriving_time, job.starting_time, job.finish_time, job.transfer_time, job.tasks[0].duration, job.nb_replicas, job.first_optimal_replica_number, job.nb_first_replicas_sended)
                    logger.debug("[%s] Master: job %s finished", self.env.now, job.job_id)

                    # Garbage-collect: a finished job never needs its data again, so free its
                    # storage on every node it was replicated to. Without this, a finished job's
                    # replicas would linger indefinitely, invisible to every future CSP solve
                    # (which only knows about jobs still being (re)scheduled) while still
                    # physically occupying space -- silently overbooking node storage.
                    for node_id in list(self.replicas_locations.get(job.job_id, [])):
                        compute_node = self.compute_nodes[node_id]
                        if job.job_id in compute_node.datasets:
                            compute_node.datasets.remove(job.job_id)
                        self.tracker.log_deletion(job.job_id, node_id, self.env.now)
                    self.replicas_locations[job.job_id] = []

            if self._allJobsCompleted():
                break

    def startTransfer(self, compute_node, job, event, duration=None):
        self.updateRunningJobs(job)

        job.replicas_nodes.append(compute_node.node_id)
        job.transfer_time = transferCost(self, job.dataset_size, compute_node.bandwidth)
        replica_inst = Replica(job.job_id, node_id=compute_node.node_id, data_size=job.dataset_size,
                                transfer_time=job.transfer_time, transfer_start_time=self.env.now)
        self.replicas_stats[(job.job_id, compute_node.node_id)] = replica_inst
        job.replicas.append(replica_inst)

        dataset_ready_event = event

        self.env.process(self.transferData(job.job_id, job.dataset_size, compute_node, dataset_ready_event, duration=duration))

        job.nb_replicas += 1
        job.node_referent = compute_node.node_id

    def getRunningJobs(self):
        """Jobs that already have a placement (i.e. not in waiting_jobs) but still have unstarted tasks."""
        to_reschedule = []
        for job in self.jobs:
            not_executed_tasks = [task for task in job.tasks if task.status == "NotStarted"]
            if job.status != "Finished" and len(not_executed_tasks) > 0:
                is_waiting = any(job.job_id == w_job.job_id for w_job in self.waiting_jobs)
                if not is_waiting:
                    to_reschedule.append(job)
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

    def updateRunningJobs(self, job):
        if job.job_id not in [j.job_id for j in self.running_jobs]:
            self.running_jobs.append(job)

    def nodesFreeTime(self, ongoing_transfers, ongoing_works):
        """
        Compute, for each compute node, how many more time units until it becomes free.

        A node has at most one ongoing transfer and one ongoing task tracked independently,
        and they can run concurrently (e.g. transferring data for job B while still executing
        a task for job A). The node is only considered free once BOTH are done, hence the
        max() below instead of letting one silently overwrite the other.
        """
        nodes_free_time = {node_id: 0 for node_id in range(len(self.compute_nodes))}

        for node_id in range(len(self.compute_nodes)):

            free_via_transfer = 0
            free_via_work = 0

            if f'node_{node_id}' in ongoing_transfers.keys() and ongoing_transfers[f'node_{node_id}'] is not None:
                _, node_id, _, t_end, duration = ongoing_transfers[f'node_{node_id}']
                free_via_transfer = int(t_end - self.env.now) + 1
                if free_via_transfer < 0:
                    logger.warning("[%s] Master: negative free time computed for node %s (ongoing transfer)", self.env.now, node_id)

            if f'node_{node_id}' in ongoing_works.keys() and ongoing_works[f'node_{node_id}'] is not None:
                job_id, node_id, k, t_start, t_end, duration = ongoing_works[f'node_{node_id}']

                task = self.jobs[job_id].tasks[k]

                if task.status == "Started":
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    free_via_work = int(t_start + execution_time - self.env.now) + 1

                elif task.status == "Finished":
                    # Same +1 margin as the other branches: this value gets truncated to an int
                    # again on the Java side (JSON has no distinct int/float, and getInt() just
                    # truncates), so leaving it as a bare float here would silently lose the
                    # fractional part with no safety margin at all.
                    free_via_work = int(task.duration * self.compute_nodes[node_id].compute_capacity) + 1

                else:
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    free_via_work = int(execution_time) + 1

            # A node is free only once BOTH its ongoing transfer and its ongoing task are done.
            nodes_free_time[node_id] = max(free_via_transfer, free_via_work)

            logger.debug("[%s] Master: node %s free in %s time unit(s)", self.env.now, node_id, nodes_free_time[node_id])

        return nodes_free_time

    def transferData(self, job_id, dataset_size, compute_node, dataset_ready_event, task_id=-1, send_task=False, duration=None):

        if job_id in self.replicas_locations.keys() and compute_node.node_id in self.replicas_locations[job_id]:
            # Data already present on this node (e.g. re-planned after the original transfer
            # already completed): nothing to transfer, just unblock whoever is waiting on it.
            # Must still mark it present in compute_node.datasets like the real-transfer path
            # below does unconditionally -- otherwise checkForJobs() in processTasks() keeps
            # returning False forever for this (node, job) pair (nothing else will ever set it,
            # since this is the only transfer this pair will ever get), and any task dispatched
            # here loops in the 0.01-tick retry queue indefinitely instead of ever starting.
            if job_id not in compute_node.datasets:
                compute_node.datasets.append(job_id)
            dataset_ready_event.succeed()
            return

        elif job_id not in self.replicas_locations.keys():
            self.replicas_locations[job_id] = []

        transfer_time = dataset_size / compute_node.bandwidth

        if self.jobs[job_id].starting_time is None:
            self.jobs[job_id].starting_time = self.env.now

        with compute_node.bandwidth_lock.request() as node_req:

            yield node_req

            logger.debug("[%s] Compute-%s: transfer of job %s dataset started, duration: %s, size: %s", self.env.now, compute_node.node_id, job_id, transfer_time, dataset_size)

            yield self.env.timeout(transfer_time)
            end_time = self.env.now

            if (compute_node.node_id, job_id) in self.dataset_events.keys() and compute_node.node_id not in self.replicas_locations[job_id]:
                self.dataset_events[(compute_node.node_id, job_id)].succeed()
                self.replicas_locations[job_id].append(compute_node.node_id)

            self.compute_nodes[compute_node.node_id].datasets.append(job_id)

            self.tracker.log_transfer(job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id)


class SchedulingUsingCSPIncremental(SchedulingUsingCSPOnline):
    """
    Comparison baseline for SchedulingUsingCSPOnline: same machinery (transfers, works,
    deletions, storage/keep-abandon handling), but the CSP is asked to place ONE
    newly-arrived job at a time, in isolation, and a job is NEVER reconsidered once
    placed -- no full replan on every arrival. Node availability accounts for whatever
    is already queued (not just the single currently-executing item), so a new job's
    plan cannot collide with work already committed to other jobs.
    """
    java_main_class = 'MainIncremental'

    def nodesFreeTimeIncremental(self, ongoing_transfers, ongoing_works):
        """Like nodesFreeTime, but also adds the backlog already queued (not yet started)
        for each node, since queued-but-undispatched work is never touched or reset here."""
        nodes_free_time = self.nodesFreeTime(ongoing_transfers, ongoing_works)
        for node_id in range(len(self.compute_nodes)):
            key = f'node_{node_id}'
            for transfer in self.transfers[key]:
                nodes_free_time[node_id] += transfer[4]  # duration
            for work in self.works[key]:
                nodes_free_time[node_id] += work[5]  # duration
        return nodes_free_time

    def _idleNodeIds(self):
        """Node ids with nothing ongoing right now (no active transfer, no active task) --
        same "free right now" notion used by the restrict_to_free_nodes hard filter in
        schedulingUsingJavaCSP. Already-queued-but-not-yet-started work doesn't disqualify a
        node: nodes_free_time already accounts for that backlog's duration."""
        idle = []
        for node_id in range(len(self.compute_nodes)):
            key = f'node_{node_id}'
            if self.ongoing_transfers.get(key) is None and self.ongoing_works.get(key) is None:
                idle.append(node_id)
        return idle

    def schedulingNewJob(self):

        # Remembers the (job_id, idle-node-set) pair that last came back with no solution when
        # restricted to free nodes, so we don't burn a full solver call re-trying the exact same
        # infeasible request every tick -- only retry once the set of idle nodes actually changes
        # (a node frees up) or a different job reaches the front of the queue. Only meaningful
        # for the restrict_to_free_nodes variant: plain Incremental sees every node's continuously
        # decreasing free-time, so a repeat solve is never truly identical to the last one there.
        last_failed_attempt = None

        while True:
            yield self.env.timeout(0.1)

            if len(self.waiting_jobs) >= 1:

                job = self.waiting_jobs[0]

                if getattr(self, 'restrict_to_free_nodes', False):
                    idle_nodes = self._idleNodeIds()
                    current_attempt = (job.job_id, frozenset(idle_nodes))
                    # An empty idle set would have schedulingUsingJavaCSP write an empty
                    # free_nodes.txt, which the Java side reads as "no restriction" (its
                    # not-empty check for opting into the filter) -- i.e. exactly the case we
                    # must not solve for would silently drop the restriction. Skip the solve
                    # entirely rather than risk that, and wait for a node to free up.
                    if not idle_nodes or current_attempt == last_failed_attempt:
                        # Nothing free, or same job/same idle nodes as the attempt that just
                        # failed: re-solving now would fail again (or silently misbehave) --
                        # wait for a node to free up instead.
                        if self._allJobsCompleted():
                            break
                        continue
                else:
                    current_attempt = None

                nodes_free_time = self.nodesFreeTimeIncremental(self.ongoing_transfers, self.ongoing_works)
                replicas_locations = self.replicas_locations

                # Incremental: place only the oldest waiting job, on its own. Jobs already
                # placed -- even if still running with unstarted tasks -- are never
                # reconsidered or re-planned.
                jobs_to_reschedule = [job]

                logger.debug("[%s] Master: looking for a solution for job %s (incremental)", self.env.now, job.job_id)
                if self._config['use_minizinc_model']:
                    transfers_, works_, deletions_ = startMinizincModel(self, jobs_to_reschedule, replicas_locations, nodes_free_time)
                else:
                    transfers_, works_, deletions_ = schedulingUsingJavaCSP(self, jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)

                if len(transfers_.keys()) > 0 and len(works_.keys()) > 0:
                    for node in range(len(self.compute_nodes)):
                        key = "node_" + str(node)
                        if key in transfers_.keys() and len(transfers_[key]) > 0:
                            for transfer in transfers_[key]:
                                self.transfers[key].append(transfer)

                            if key in works_.keys() and len(works_[key]) > 0:
                                for work in works_[key]:
                                    self.works[key].append(work)

                        if key in deletions_.keys() and len(deletions_[key]) > 0:
                            for deletion in deletions_[key]:
                                self.deletions[key].append(deletion)

                    self.waiting_jobs.pop(0)
                    last_failed_attempt = None
                else:
                    logger.warning("[%s] Master: no CSP solution found for job %s, will retry", self.env.now, job.job_id)
                    last_failed_attempt = current_attempt

            if self._allJobsCompleted():
                break


class SchedulingUsingCSPIncrementalFreeNodesOnly(SchedulingUsingCSPIncremental):
    """
    Same as SchedulingUsingCSPIncremental (one job at a time, never reconsidered), but the CSP
    is only allowed to pick among nodes with nothing ongoing RIGHT NOW (no active transfer, no
    active task) -- a hard filter, with no notion of when a currently-active node would become
    free. A node with only already-queued-but-not-yet-started future work is still a candidate
    (nodes_free_time already accounts for that backlog's duration, so it won't be double-booked).
    A node actively busy right now is simply not a candidate for this solve, period (still also
    subject to the existing storage-size filter).
    """
    restrict_to_free_nodes = True


class SchedulingUsingCSPSemiOnline:
    
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
        self.actual_transfers = {}
        self.overlap = self._config['overlap'] if 'overlap' in self._config else overlap
        self.replicas_locations = {}
        self.threshold = config['threshold']
        self.dataset_sizes = []
        self.jobs = []
        self.ongoing_transfers = {}
        self.ongoing_works = {}
        self.transfers = {}        
        self.works = {}
        
        
        logging.debug(f"Master node with {self.nb_nodes} compute nodes")

    def receiveJobs(self,):

        while True:
            #logger.debug("[%s] Master: Waiting for new job", self.env.now)
            new_job = yield self.queue.get()
            new_job.arriving_time = self.env.now

            #logger.debug("[%s] Master: Job %s Arrived.", self.env.now, new_job.job_id)

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
    
    def schedulingNewJob(self,):
        for node_id in range(len(self.compute_nodes)):
            self.ongoing_transfers[f'node_{node_id}'] = None
            self.ongoing_works[f'node_{node_id}'] = None
            self.transfers[f'node_{node_id}'] = []
            self.works[f'node_{node_id}'] = []

        while True:

            yield self.env.timeout(0.2)

            if len(self.waiting_jobs) >= 1 and len(self.nodesFree(self.ongoing_transfers, self.ongoing_works, self.works, self.transfers, self.waiting_jobs[0] ) ) > 0: #0 and self.isNoJobRunning(self.waiting_jobs[0].job_id, transfers, works): #self.env.now - t_now > 600: # and len(self.waiting_jobs + self.jobsToReschedule()) > 0: #len(self.waiting_jobs) > 0 and not block:

                    free_nodes_list = self.nodesFree(self.ongoing_transfers, self.ongoing_works, self.works, self.transfers, self.waiting_jobs[0] )
                    replicas_locations = self.replicas_locations
                    jobs_to_reschedule = copy.deepcopy(self.waiting_jobs) # + self.jobsToReschedule()
                    print("waiting jobs:", [job.job_id for job in self.waiting_jobs])
                    print(len(jobs_to_reschedule), ' jobs to reschedule at time ', self.env.now)

                    if len(jobs_to_reschedule) > 0 and len(free_nodes_list.keys()) > 0:
                        logger.debug("[%s] Master: Start looking for a solution. at time %s", self.env.now,self.env.now)

                        jobs = [jobs_to_reschedule[0]]
                        print("jobs to reschedule:", [job.job_id for job in jobs])
                        free_nodes = [self.compute_nodes[node_c] for node_c in free_nodes_list.keys()]
                        nodes_free_time = [free_nodes_list[node_c] for node_c in free_nodes_list.keys()]
                        transfer_node_free_time = [0 for node_c in free_nodes_list.keys()]

                        replicas_locations = {0:[]}
                                                                 #   master_node, jobs: list, r        eplicas_locations: dict, nodes_free_time: list, scheduling_start_time=None):
                        transfers_, works_ = schedulingUsingJavaCSP(self,jobs,    replicas_locations, free_nodes,            nodes_free_time,        transfer_node_free_time)

                        node_to_use = []
                        for i, node_used in enumerate(transfers_.keys()):
                            if len(transfers_[node_used]) > 0:
                                node_to_use.append(free_nodes[int(node_used[5:])].node_id) #, self.compute_nodes[free_nodes_list[int(node_used[5:])]])

                    else:
                        transfers_, works_ = {}, {}

                    if len(node_to_use) > 0: #len(transfers_.keys()) > 0 and len(works_.keys()) > 0:
                        
                        for node in node_to_use:#key in transfers_.keys():
                            key = "node_"+str(node)
                            
                            if key in transfers_.keys() and len(transfers_[key]) > 0:
                                for k in range(len(transfers_[key])):
                                    (job_id, _, t_start, t_end, duration) = transfers_[key][k]
                                    t_start += self.env.now
                                    t_end += self.env.now
                                    self.transfers[key].append((job_id, _, t_start, t_end, duration))
                                
                                if len(works_[key]) > 0:
                                    for k in range(len(works_[key])):
                                        (job_id, _, k, t_start, t_end, duration) = works_[key][k]
                                        t_start += self.env.now
                                        t_end += self.env.now
                                        self.works[key].append((job_id, _, k, t_start, t_end, duration))
                        #print(transfers, works)
                        #scheduling_start_time = int(self.env.now)
                        
                        l = len(self.waiting_jobs)
                        for i in range(1): 
                            if len(self.waiting_jobs) > 0:
                                self.waiting_jobs.pop(0)

                        
            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.jobs) == self._config['total_nb_jobs']:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break
    
    def scheduling(self):
    
        
        self.dataset_events = {}
        
        
        while True:

            yield self.env.timeout(1)

            if not self.transfers and not self.works:
                continue
            
            for node_id in range(len(self.compute_nodes)):  

                if self.ongoing_transfers[f'node_{node_id}'] is not None:
                    job_id, _, t_start, _, duration =  self.ongoing_transfers[f'node_{node_id}']
                    if self.env.now >=  t_start + duration: #transferCost(self,self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth):
                        self.ongoing_transfers[f'node_{node_id}'] = None

                if f'node_{node_id}' in self.transfers.keys() and len(self.transfers[f'node_{node_id}']) > 0 and self.ongoing_transfers[f'node_{node_id}'] is None:
                    
                    (job_id, _, t_start, _, duration) = self.transfers[f'node_{node_id}'][0]
                    if t_start <= self.env.now:
                        (job_id, _, t_start, _, duration) = self.transfers[f'node_{node_id}'].pop(0)
                        transfer_time = transferCost(self, self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth, self._config)
                        t_s = int(self.env.now)+1
                        self.ongoing_transfers[f'node_{node_id}'] = (job_id, node_id, t_s, t_s + transfer_time-0.2, transfer_time)
                        self.dataset_events[(node_id, job_id)] = self.env.event()
                        self.startTransfer(self.compute_nodes[node_id], self.jobs[job_id],self.dataset_events[(node_id, job_id)])
                            

                if self.ongoing_works[f'node_{node_id}'] is not None:
                    job_id, _, k, t_start, t_end, d =  self.ongoing_works[f'node_{node_id}']
                    task = self.jobs[job_id].tasks[k]
                    if task.status == "Finished":
                        self.ongoing_works[f'node_{node_id}'] = None
                            
                if f'node_{node_id}' in self.works.keys() and len(self.works[f'node_{node_id}']) > 0 and self.ongoing_works[f'node_{node_id}'] is None:


                    (job_id, _,k, t_start, _, duration) = self.works[f'node_{node_id}'][0]

                    if t_start <= self.env.now and (node_id, job_id) in self.dataset_events.keys():
                        (job_id, id_node,k, t_start, t_end, duration) = self.works[f'node_{node_id}'].pop(0)
                        
                        job = self.jobs[job_id]
                        compute_node = self.compute_nodes[node_id]

                        #not_executed_tasks = [task for task in job.tasks if task.status == "NotStarted"]
                        task = self.jobs[job_id].tasks[k] #None if len(not_executed_tasks) == 0 else not_executed_tasks[0]
                        
                        if task:
                            task.dataset_ready_event = self.dataset_events[(node_id, job_id)]
                            job.nb_remaining_tasks -= 1
                            ts = self.env.now
                            ex_time = task.duration * compute_node.compute_capacity
                            self.ongoing_works[f'node_{node_id}'] = (job_id, node_id, task.task_id, ts, ts+ex_time, ex_time)
                            task.node = compute_node.node_id
                            self.replicas_stats[(job.job_id, task.node)].nb_tasks +=1
                            self.replicas_stats[(job.job_id, task.node)].task_execution_time += ex_time
                            task.status = "Scheduled"
                            
                            yield compute_node.queue.put(task)

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
                            #logger.debug("[%s] Master: Task %s of job %s finished on node %s.", self.env.now, task.task_id, job.job_id, task.node)
                            #self.endTask(job.job_id)    

            for job in self.jobs:
               finished_tasks = [task for task in job.tasks if task.status == "Finished"]
               if len(finished_tasks) == len(job.tasks) and job.status!="Finished":
                   self.finished_jobs += 1
                   job.finish_time = np.max([task.finishing_time for task in job.tasks])
                   job.status = "Finished"
                   self.tracker.log_end_job(job.job_id,len(job.tasks),job.dataset_size,job.arriving_time,job.starting_time, job.finish_time,job.transfer_time, job.tasks[0].duration, job.nb_replicas, job.first_optimal_replica_number,job.nb_first_replicas_sended)
                   logger.debug("[%s] Master: Job %s finished.", self.env.now, job.job_id)

            #print(self.finished_jobs, self._config['total_nb_jobs'] ,len(self.waiting_jobs) == 0 ,len(self.jobs) == self._config['total_nb_jobs'])
            #print(self.finished_jobs == self._config['total_nb_jobs'],  len(self.waiting_jobs) == 0,  len(self.jobs) == self._config['total_nb_jobs'])
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

    def jobsToReschedule(self):
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
                #nodes_free_time[node_id] = (scheduling_start_time + t_end) - self.env.now
                #execution_time = self.jobs.tasks[0].duration
                nodes_free_time[node_id] = int(t_end - self.env.now)
                #nodes_free_time[node_id] = (scheduling_start_time + t_start + execution_time + execution_time*self.compute_nodes[node_id].compute_capacity ) - self.env.now
            
            if f'node_{node_id}' in ongoing_works.keys() and ongoing_works[f'node_{node_id}'] is not None:
                #(job_id, node_id,k, t_start, t_end, duration)
                job_id, node_id, k, t_start, t_end, duration =  ongoing_works[f'node_{node_id}']
                #transfer_time = transferCost(self,self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth, self._config)
                task = self.jobs[job_id].tasks[k]
                if task.status == "Started":
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = int(task.starting_time + execution_time- self.env.now)
                if task.status == "Finished":
                    nodes_free_time[node_id] = 0
                else:
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = execution_time

            if ongoing_works[f'node_{node_id}'] is None or ongoing_transfers[f'node_{node_id}'] is None:
                nodes_free_time[node_id] = 0
        return nodes_free_time
    
    def nodesFree(self, ongoing_transfers, ongoing_works, transfers, works, job):
        nodes_free_time = {node_id: 0 for node_id in range(len(self.compute_nodes))}
        transfer_node_free_time = {}
        
        for node_id in range(len(self.compute_nodes)):

            if f'node_{node_id}' in ongoing_transfers.keys() and ongoing_transfers[f'node_{node_id}'] is not None:
                _, node_id, _, t_end, duration =  ongoing_transfers[f'node_{node_id}'] 
                nodes_free_time[node_id] = int(t_end - self.env.now)
                #transfer_node_free_time[node_id] = int(t_end - self.env.now)

            if f'node_{node_id}' in ongoing_works.keys() and ongoing_works[f'node_{node_id}'] is not None:
                job_id, node_id, k, t_start, t_end, duration =  ongoing_works[f'node_{node_id}']
                task = self.jobs[job_id].tasks[k]

                if task.status == "Started":
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = int(t_start + execution_time - self.env.now)
                    #transfer_node_free_time[node_id] = transferCost(self,self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth, self._config) - int(t_end - self.env.now)

                elif task.status == "Finished":
                    nodes_free_time[node_id] = 0.1
                    #transfer_node_free_time[node_id] = 0

                else:
                    execution_time = task.duration * self.compute_nodes[node_id].compute_capacity
                    nodes_free_time[node_id] = execution_time
                    #transfer_node_free_time[node_id] = transferCost(self,self.jobs[job_id].dataset_size, self.compute_nodes[node_id].bandwidth, self._config) - duration
            
            for transfer in self.transfers[f'node_{node_id}']:
                job_id, _, t_start, t_end, duration = transfer
                nodes_free_time[node_id] += self.jobs[job_id].dataset_size / self.compute_nodes[node_id].bandwidth
                #transfer_node_free_time[node_id] += duration
                
            for work in self.works[f'node_{node_id}']:
                job_id, _,k, t_start, t_end, duration = work
                nodes_free_time[node_id] += self.jobs[job_id].tasks[0].duration * self.compute_nodes[node_id].compute_capacity
                #transfer_node_free_time[node_id] += duration
            
        return nodes_free_time
    
    def transferData(self, job_id, dataset_size, compute_node, dataset_ready_event,task_id= -1, send_task = False):

        if job_id in self.replicas_locations.keys() and compute_node.node_id in self.replicas_locations[job_id]:
            dataset_ready_event.succeed()
            return

        elif job_id not in self.replicas_locations.keys():
            self.replicas_locations[job_id] = []

        self.replicas_locations[job_id].append(compute_node.node_id)

        transfer_time = dataset_size / compute_node.bandwidth
        self.actual_transfers[self.compute_nodes[compute_node.node_id].node_id] = (job_id, self.env.now + transfer_time)

        if self.jobs[job_id].starting_time is None: self.jobs[job_id].starting_time = self.env.now

        with compute_node.bandwidth_lock.request() as node_req:
            self.compute_nodes[compute_node.node_id].job_order.append(job_id)
            logger.debug("[%s] Compute-%s: Got new dataset transfert of job %s: duration: %s, dataset_size: %s",
                         self.env.now, compute_node.node_id, job_id, transfer_time, dataset_size)
            yield node_req  
            
            self.compute_nodes[compute_node.node_id].running_task.append(-1)
            
            yield self.env.timeout(transfer_time)

            end_time = self.env.now
            
            self.compute_nodes[compute_node.node_id].running_task.pop(0)
            self.compute_nodes[compute_node.node_id].datasets.append(job_id)
            
            dataset_ready_event.succeed()

            self.tracker.log_transfer(job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id)

            if compute_node.node_id in self.actual_transfers.keys(): del self.actual_transfers[self.compute_nodes[compute_node.node_id].node_id]

