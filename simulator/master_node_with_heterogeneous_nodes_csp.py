import json
import math
import os
import random
import simpy
import time
import threading
import concurrent.futures
import numpy as np
import logging
from classes.job import Task, Replica, Job
import copy
from compute_node import ComputeNode

from utils.modelCSP import onLineSchedulingUsingCSP,schedulingUsingJavaCSP,startMinizincModel,MODEL_DIR,SIMULATOR_DIR,run_cwd_override
from utils.run_export import solver_model_dir


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
    # Whether a CSP solve's own real wall-clock time is charged against simulated time (yield
    # env.timeout(elapsed)) right after it returns -- modeling "the system must wait for the
    # solver to actually finish" as a real cost against every job's own flow time, uniformly
    # across every approach (see _timedSchedulingUsingJavaCSP below). True is the FAIR default:
    # before this existed, only SchedulingUsingCSPAdaptiveJoint ever charged anything (its own
    # escalation budget, an estimate, not the real solve time), while online/online_biobj/
    # incremental charged nothing at all -- a structural asymmetry that penalized hybrid in any
    # flow-time comparison regardless of solution quality (confirmed: disabling hybrid's own
    # charge alone made its very first job's result land EXACTLY on online_biobj's own result).
    # Set to False to reproduce the old (asymmetric) behavior for a specific approach/run.
    charge_thinking_time = True

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

    def _timedSchedulingUsingJavaCSP(self, jobs_to_reschedule, replicas_locations, nodes_free_time, now):
        """Generator wrapper around schedulingUsingJavaCSP: times the real wall-clock cost of the
        solve and (if charge_thinking_time) yields that same duration against simulated time
        before returning -- see the charge_thinking_time class attribute's own comment for why.
        Callers must use `yield from` (this is a generator itself, not a plain function)."""
        start = time.time()
        transfers_, works_, deletions_ = schedulingUsingJavaCSP(self, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
        elapsed = time.time() - start
        charge = self._config.get('charge_thinking_time', self.charge_thinking_time)
        if charge:
            yield self.env.timeout(elapsed)
        return transfers_, works_, deletions_

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
                        transfers_, works_, deletions_ = yield from self._timedSchedulingUsingJavaCSP(jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)
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

                # Bug found 2026-09-28: this used to free the slot on a naive ESTIMATE
                # (env.now >= t_start + duration) instead of the transfer's REAL completion.
                # transferData() must first acquire compute_node.bandwidth_lock (a capacity-
                # limited simpy.Resource -- several transfers can be genuinely concurrent up to
                # that capacity), so under contention the real transfer can finish LATER than
                # this estimate assumed. The slot then looked "free" while the old transfer was
                # still in flight, letting scheduling() dispatch a SECOND transfer for the same
                # (node, job) pair -- confirmed via a live crash: "Event ... has already been
                # triggered" in transferData, from two dispatches racing to succeed() the same
                # dataset_events entry (and, when timing differs slightly, an orphaned event
                # instead of a crash -- the task waiting on it then never starts, i.e. the job
                # never finishes). Freeing the slot is now transferData's own job, done exactly
                # once, right when ITS transfer genuinely completes -- see its own comment.
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
                    job_id, _, k, _, exp_end, _ = self.ongoing_works[f'node_{node_id}']
                    task = self.jobs[job_id].tasks[k]
                    if task.status == "Finished":
                        self.ongoing_works[f'node_{node_id}'] = None
                    elif self.env.now > exp_end + 1:
                        # Diagnostic added 2026-09-28: this node's "ongoing" slot is occupied by a
                        # task whose own expected end time is long past, yet task.status never
                        # reached "Finished" -- blocks this node from ever dispatching its NEXT
                        # queued works entry (line ~272's own gate requires ongoing_works is None),
                        # a second possible head-of-line-blocking mechanism alongside the
                        # dataset_events one below. Throttled.
                        stall_key = ('ongoing_works', node_id, job_id, k)
                        stall_counts = getattr(self, '_dispatch_stall_counts', None)
                        if stall_counts is None:
                            stall_counts = {}
                            self._dispatch_stall_counts = stall_counts
                        stall_counts[stall_key] = stall_counts.get(stall_key, 0) + 1
                        if stall_counts[stall_key] in (1, 100, 10000, 1000000):
                            print(f"### DIAG ongoing_works STALLED: node_{node_id} job {job_id} task_k={k} "
                                  f"status={task.status!r} expected_end={exp_end:.2f} now={self.env.now:.2f} "
                                  f"stall count={stall_counts[stall_key]} ###", flush=True)

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

                    ready = t_start <= self.env.now
                    has_event = (node_id, job_id) in self.dataset_events.keys()
                    if ready and not has_event:
                        # Diagnostic added 2026-09-28: investigating jobs that never finish at
                        # larger scale (20j/50n) -- head-of-line blocking suspected: a works entry
                        # stuck here forever (t_start long past, but no dataset_events key for this
                        # (node,job) pair) blocks every OTHER works entry queued behind it on this
                        # same node too, even for unrelated jobs. Throttled to avoid flooding.
                        stall_key = (node_id, job_id)
                        stall_counts = getattr(self, '_dispatch_stall_counts', None)
                        if stall_counts is None:
                            stall_counts = {}
                            self._dispatch_stall_counts = stall_counts
                        stall_counts[stall_key] = stall_counts.get(stall_key, 0) + 1
                        if stall_counts[stall_key] in (1, 100, 10000, 1000000):
                            print(f"### DIAG works dispatch STALLED: node_{node_id} job {job_id} task_k={k} "
                                  f"t_start={t_start:.2f} now={self.env.now:.2f} -- no dataset_events "
                                  f"entry for (node_{node_id}, job {job_id}) -- stall count="
                                  f"{stall_counts[stall_key]}, queue_len={len(self.works[f'node_{node_id}'])}, "
                                  f"replicas_locations[{job_id}]={self.replicas_locations.get(job_id)} ###",
                                  flush=True)

                    if ready and has_event:
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

        # Bug found 2026-09-28: ongoing_transfers[node_id] (the "this node's transfer slot is
        # busy" flag) used to be freed by scheduling() itself on a naive time ESTIMATE, before
        # this coroutine's real transfer necessarily finished (see scheduling()'s own comment on
        # this) -- freeing it here instead, exactly once, right when THIS transfer genuinely
        # completes, closes that race. Guarded on job_id matching in case some other, unrelated
        # bug ever leaves a mismatched entry there -- never clear a slot this call doesn't own.
        def _freeOwnSlot():
            key = f'node_{compute_node.node_id}'
            current = self.ongoing_transfers.get(key)
            if current is not None and current[0] == job_id:
                self.ongoing_transfers[key] = None

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
            _freeOwnSlot()
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
            _freeOwnSlot()

            # Succeed the event THIS call was actually asked to signal (the parameter), not
            # whatever schedulingUsingJavaCSP's dispatch loop currently has in self.dataset_events
            # for this (node, job) key -- that dict entry can only ever be reassigned to a NEW
            # event now that the race above is closed, but reading through the parameter instead
            # of the dict is the more directly correct semantic regardless, and was the actual
            # crash site (double-succeed()) when the race above still existed.
            if compute_node.node_id not in self.replicas_locations[job_id]:
                dataset_ready_event.succeed()
                self.replicas_locations[job_id].append(compute_node.node_id)

            self.compute_nodes[compute_node.node_id].datasets.append(job_id)

            self.tracker.log_transfer(
                job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id,
                receiver_energy_consumption=compute_node.energy_consumption,
                sender_energy_consumption=self._config.get('master_energy_consumption', 0.0),
                network_energy=self._config.get('network_energy_per_transfer', 0.0),
            )


class SchedulingUsingCSPOnlineMultiObj(SchedulingUsingCSPOnline):
    """
    Same full-replan Online approach as SchedulingUsingCSPOnline, but each replan solves the
    bi-objective epsilon-constraint problem via MainOnlineMultiObj.java instead of MainOnline.java:
    phase 1 minimizes max flow time (identical objective to plain Online), phase 2 then minimizes
    total transfer energy subject to max flow time staying within epsilon_fraction of phase 1's
    result (optionally clamped by epsilon_max_cap). See utils/modelCSP.py's schedulingUsingJavaCSP
    for how the class attributes below get forwarded to the Java model, and
    utils/model/src/main/MainOnlineMultiObj.java for the 2-phase solve itself.
    """
    java_main_class = 'MainOnlineMultiObj'
    multi_objective = 2  # epsilon-constraint (2-phase): phase 1 = max flow time, phase 2 = energy
    epsilon_fraction = 0.1
    epsilon_phase1_fraction = 0.5
    epsilon_max_cap = None


class SchedulingUsingCSPOnlineWarmStart(SchedulingUsingCSPOnline):
    """
    Hybrid experiment: same full-replan Online approach, but seeds the CSP's search with a
    warm-start solution instead of letting it start cold every time. The warm start combines:
      (a) this scheduler's OWN last-known plan for jobs it already knew about (read back from
          self.works/self.transfers, exactly what's "currently running/queued" from Online's own
          point of view -- no attempt to reconcile with a different scheduler's timeline), and
      (b) Incremental's decision for the brand-new job(s), computed via a throwaway MainIncremental
          solve using the EXACT SAME nodes_free_time/replicas_locations this replan itself sees
          (so it's state-consistent with Online, not with Incremental's own separate history).
    Tests whether Online's usual underperformance vs. Incremental comes mainly from its LNS
    search not reliably reaching a good solution within budget (in which case warm-starting from
    an at-least-as-good point should close most of the gap) rather than full replanning being
    inherently worse.
    """
    java_main_class = 'MainOnlineWarmStart'

    def schedulingNewJob(self):

        while True:
            yield self.env.timeout(0.1)

            if len(self.waiting_jobs) >= 1:

                nodes_free_time = self.nodesFreeTime(self.ongoing_transfers, self.ongoing_works)
                replicas_locations = self.replicas_locations
                jobs_to_reschedule = [job for job in self.waiting_jobs] + self.getRunningJobs()

                if len(jobs_to_reschedule) > 0:
                    self._writeWarmStart(jobs_to_reschedule, replicas_locations, nodes_free_time)
                    logger.debug("[%s] Master: looking for a solution for %s job(s) (warm-started)", self.env.now, len(jobs_to_reschedule))
                    transfers_, works_, deletions_ = yield from self._timedSchedulingUsingJavaCSP(jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)
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

                    self.waiting_jobs.clear()

                else:
                    logger.warning("[%s] Master: no CSP solution found for %s job(s)", self.env.now, len(jobs_to_reschedule))

            if self._allJobsCompleted():
                break

    def _writeWarmStart(self, jobs_to_reschedule, replicas_locations, nodes_free_time):
        """Builds warm_start.json for the upcoming solve: this scheduler's own last-decided plan
        for jobs it already knew about, plus Incremental's decision for the brand-new job(s),
        both converted from absolute simulation time to the local (relative-to-now) time frame
        this solve's Java model expects (mirrors how toDict() adds `now` back on the way out)."""
        sorted_jobs = sorted(jobs_to_reschedule, key=lambda j: j.job_id)
        job_index = {job.job_id: idx for idx, job in enumerate(sorted_jobs)}
        now = self.env.now

        job_placements = []
        transfers_ws = []

        # (a) already-known jobs: reuse this scheduler's own last-decided plan as-is.
        already_known_ids = {job.job_id for job in self.getRunningJobs()}
        for node_id in range(len(self.compute_nodes)):
            key = f'node_{node_id}'
            for work in self.works.get(key, []):
                w_job_id, w_node, w_task, w_start, w_end, w_dur = work
                if w_job_id in job_index and w_job_id in already_known_ids:
                    job_placements.append({
                        "job_index": job_index[w_job_id], "task_index": int(w_task),
                        "node": int(w_node), "start": int(round(w_start - now)),
                    })
            for transfer in self.transfers.get(key, []):
                t_job_id, t_node, t_start, t_end, t_dur = transfer
                if t_job_id in job_index and t_job_id in already_known_ids:
                    transfers_ws.append({
                        "job_index": job_index[t_job_id], "node": int(t_node),
                        "start": int(round(t_start - now)),
                    })

        # (b) brand-new job(s): ask Incremental, using the exact state this replan itself sees.
        orig_java_main_class = self.java_main_class
        try:
            for job in self.waiting_jobs:
                self.java_main_class = 'MainIncremental'
                inc_transfers, inc_works, _ = schedulingUsingJavaCSP(
                    self, [job], replicas_locations, nodes_free_time, self.env.now)
                for node_id in range(len(self.compute_nodes)):
                    key = f'node_{node_id}'
                    for work in inc_works.get(key, []):
                        w_job_id, w_node, w_task, w_start, w_end, w_dur = work
                        if w_job_id == job.job_id:
                            job_placements.append({
                                "job_index": job_index[w_job_id], "task_index": int(w_task),
                                "node": int(w_node), "start": int(round(w_start - now)),
                            })
                    for transfer in inc_transfers.get(key, []):
                        t_job_id, t_node, t_start, t_end, t_dur = transfer
                        if t_job_id == job.job_id:
                            transfers_ws.append({
                                "job_index": job_index[t_job_id], "node": int(t_node),
                                "start": int(round(t_start - now)),
                            })
        finally:
            self.java_main_class = orig_java_main_class

        warm_start = {"job_placements": job_placements, "transfers": transfers_ws}
        # solver_model_dir() (not the fixed MODEL_DIR) so this lands where the solve about to run
        # will actually read it from -- under SIMULATOR_RUN_CWD isolation (e.g. the Grid5000
        # submission scripts, which give each approach its own private utils/model tree so
        # concurrent runs don't clobber each other's exchange files) MODEL_DIR is the wrong,
        # shared location: the JVM reads warm_start.json relative to ITS OWN cwd (the isolated
        # one), so a write to MODEL_DIR here was silently invisible to it, and warm-starting
        # never actually applied even though this call appeared to succeed.
        with open(os.path.join(solver_model_dir(), "inputs", "warm_start.json"), "w") as f:
            json.dump(warm_start, f)


class SchedulingUsingCSPOnlineMultiObjWarmStart(SchedulingUsingCSPOnlineWarmStart):
    """
    Combines SchedulingUsingCSPOnlineMultiObj's epsilon-constraint bi-objective solve (phase 1 =
    max flow time, phase 2 = transfer energy within epsilon_fraction of phase 1's result) with
    SchedulingUsingCSPOnlineWarmStart's warm-starting (this scheduler's own last-known plan for
    already-known jobs + a throwaway Incremental solve for the brand-new job(s)) -- the
    schedulingNewJob()/_writeWarmStart() machinery is inherited as-is from
    SchedulingUsingCSPOnlineWarmStart, only java_main_class and the epsilon settings differ.

    See utils/model/src/main/MainOnlineMultiObjWarmStart.java for the Java side: it seeds ONLY
    the shared search's value selector with the external warm start (Incremental's hint), and
    re-points that same value selector at model.getSolver().defaultSolution() right after phase 1
    ends so phase 2 still inherits phase 1's own result -- restoring plain
    MainOnlineMultiObj.java's phase1->phase2 warm-start property that would otherwise be lost by
    overriding the search strategy's hint for the external warm start.
    """
    java_main_class = 'MainOnlineMultiObjWarmStart'
    multi_objective = 2  # epsilon-constraint (2-phase): phase 1 = max flow time, phase 2 = energy
    epsilon_fraction = 0.1
    epsilon_phase1_fraction = 0.5
    epsilon_max_cap = None


class SchedulingUsingCSPOnlineThreeStep(SchedulingUsingCSPOnline):
    """
    Same online full-replan machinery as SchedulingUsingCSPOnline, but delegates the
    CSP solve to the three-step decomposition (Dataset Splitting -> per-node Transfer
    scheduling -> per-node Task Allocation) adapted from flowtime-scheduler-main,
    instead of the single-shot Choco model used by MainOnline.java.
    """
    java_main_class = 'MainOnlineThreeStep'


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
                    transfers_, works_, deletions_ = yield from self._timedSchedulingUsingJavaCSP(jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)

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


class SchedulingUsingCSPAdaptive(SchedulingUsingCSPIncremental):
    """
    Incremental-first escalation: each new job is placed by Incremental (MainIncremental) first
    to get a fast, near-free flow-time estimate F1. If escalating looks worth it, a second,
    slower search (MainOnlineMultiObj by default) is given a budget of `adaptive_alpha * F1`
    seconds to try to beat it for this SAME single job -- and only that job: like Incremental,
    already-running jobs are never reconsidered here.

    Two things this deliberately gets right, both learned the hard way earlier in this project:
      - The escalation search's own decision time is not free. Its infrastructure snapshot
        (nodes_free_time, replicas_locations) is taken only AFTER waiting out the budget --
        i.e. at env.now == T + budget, not at the job's arrival T -- via a plain
        `yield self.env.timeout(budget)` before the second solve, which lets every other SimPy
        process (scheduling(), other arrivals, ongoing work finishing) advance for real during
        that wait. schedulingUsingJavaCSP is then called with scheduling_start_time=env.now
        (now T + budget), so toDict() anchors the escalation plan's absolute start/end times
        there -- its flow time F2 (finish - job.arriving_time) already reflects the wait, with
        no separate bookkeeping needed.
      - Ongoing tasks/transfers are never touched or re-decided (same invariant as every other
        approach here): the job stays in self.waiting_jobs, undispatched, for the whole budget
        window, and only the one job's own placement is ever at stake -- nothing already
        committed to other jobs is reopened.
      - F1 itself is only ever used as a *prediction* to size the budget and as the baseline for
        the gain check below -- by the time a decision is committed (whichever way it goes), the
        world has moved on, so the final placement (Incremental fallback included) is always
        computed fresh against the current state, never replayed from a now-stale probe.

    Escalation is accepted only if it clears `adaptive_alpha` as a *relative gain* over F1:
    (F1 - F2) / F1 > adaptive_alpha. Same knob for both the search budget and the acceptance bar
    by design (one tunable parameter, not two) -- a job with F1 too small to be worth chasing
    also gets too small a budget to plausibly clear the bar, so no separate "is F1 worth it?"
    gate is needed on top.

    The budget is capped at `adaptive_max_budget_s` (default 1200s = 20min): with short task
    durations, alpha * F1 can be too small for the Java/Choco side (JVM startup alone eats a
    real chunk of a very short budget) to get anywhere near a meaningful search on a 2-phase
    epsilon-constraint problem -- so raise `adaptive_alpha` to get real solver time out of small
    jobs too, and this cap keeps the rare very-large-F1 job from then running unboundedly long.
    """
    java_main_class = 'MainIncremental'
    escalation_java_main_class = 'MainOnlineMultiObj'
    adaptive_alpha = 0.2
    adaptive_max_budget_s = 1200

    def _flowTimeFromPlan(self, job, transfers_, works_):
        """Flow time (finish - arrival) this job would realize under a given plan, from the
        plan's own absolute start/end times (already anchored to whatever env.now was when the
        solve that produced it ran -- see toDict() in utils/modelCSP.py). None if the job doesn't
        appear in the plan at all (infeasible/empty solve)."""
        end_times = []
        for entries in transfers_.values():
            for t_job_id, t_node, t_start, t_end, t_dur in entries:
                if t_job_id == job.job_id:
                    end_times.append(t_end)
        for entries in works_.values():
            for w_job_id, w_node, w_task, w_start, w_end, w_dur in entries:
                if w_job_id == job.job_id:
                    end_times.append(w_end)
        if not end_times:
            return None
        return max(end_times) - job.arriving_time

    def _placeSingleJob(self, job, nodes_free_time, java_main_class, solver_time_limit_s=None):
        """One throwaway solve placing only `job`, using `java_main_class` -- exactly the same
        request shape Incremental itself uses (see schedulingNewJob below), just with the main
        class and (optionally) the solver time budget swapped out for the call and restored
        right after, mirroring the pattern already used by SchedulingUsingCSPOnlineWarmStart."""
        orig_java_main_class = self.java_main_class
        orig_limit = self._config.get('solver_time_limit_s')
        try:
            self.java_main_class = java_main_class
            if solver_time_limit_s is not None:
                self._config['solver_time_limit_s'] = max(1, int(round(solver_time_limit_s)))
            transfers_, works_, deletions_ = schedulingUsingJavaCSP(
                self, [job], self.replicas_locations, nodes_free_time, self.env.now)
        finally:
            self.java_main_class = orig_java_main_class
            if solver_time_limit_s is not None:
                if orig_limit is None:
                    self._config.pop('solver_time_limit_s', None)
                else:
                    self._config['solver_time_limit_s'] = orig_limit
        flow_time = self._flowTimeFromPlan(job, transfers_, works_)
        return transfers_, works_, deletions_, flow_time

    def schedulingNewJob(self):

        while True:
            yield self.env.timeout(0.1)

            if len(self.waiting_jobs) >= 1:

                job = self.waiting_jobs[0]

                nodes_free_time = self.nodesFreeTimeIncremental(self.ongoing_transfers, self.ongoing_works)
                logger.debug("[%s] Master: probing Incremental for job %s (adaptive)", self.env.now, job.job_id)
                transfers_, works_, deletions_, f1 = self._placeSingleJob(job, nodes_free_time, 'MainIncremental')

                if f1 is None:
                    logger.warning("[%s] Master: no CSP solution found for job %s (adaptive/incremental probe), will retry", self.env.now, job.job_id)
                    if self._allJobsCompleted():
                        break
                    continue

                alpha = self._config.get('adaptive_alpha', self.adaptive_alpha)
                max_budget = self._config.get('adaptive_max_budget_s', self.adaptive_max_budget_s)
                budget = min(alpha * f1, max_budget)

                if budget > 0:
                    logger.debug("[%s] Master: job %s F1=%.3f -> escalation budget=%.3f (snapshot at %.3f)",
                                 self.env.now, job.job_id, f1, budget, self.env.now + budget)
                    yield self.env.timeout(budget)

                    nodes_free_time_2 = self.nodesFreeTimeIncremental(self.ongoing_transfers, self.ongoing_works)
                    _, _, _, f2 = self._placeSingleJob(job, nodes_free_time_2, self.escalation_java_main_class,
                                                        solver_time_limit_s=budget)

                    gain = (f1 - f2) / f1 if f2 is not None else -1.0

                    if f2 is not None and gain > alpha:
                        logger.debug("[%s] Master: job %s escalation accepted (F1=%.3f -> F2=%.3f, gain=%.1f%%)",
                                     self.env.now, job.job_id, f1, f2, gain * 100)
                        # Re-solve one last time instead of reusing the just-computed plan: the
                        # gain check itself takes zero extra time, but keeping the escalation
                        # solve and the commit as two logically separate steps means a future
                        # change adding any delay in between can't silently commit a stale plan.
                        nodes_free_time_final = self.nodesFreeTimeIncremental(self.ongoing_transfers, self.ongoing_works)
                        transfers_, works_, deletions_, _ = self._placeSingleJob(
                            job, nodes_free_time_final, self.escalation_java_main_class, solver_time_limit_s=budget)
                    else:
                        logger.debug("[%s] Master: job %s escalation rejected (F1=%.3f, F2=%s), falling back to Incremental",
                                     self.env.now, job.job_id, f1, f2)
                        # State moved on during the wait -- re-probe Incremental fresh rather
                        # than committing the now-stale pre-wait plan.
                        nodes_free_time_fallback = self.nodesFreeTimeIncremental(self.ongoing_transfers, self.ongoing_works)
                        transfers_, works_, deletions_, _ = self._placeSingleJob(
                            job, nodes_free_time_fallback, 'MainIncremental')

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
                else:
                    logger.warning("[%s] Master: no CSP solution found for job %s (adaptive final), will retry", self.env.now, job.job_id)

            if self._allJobsCompleted():
                break


class SchedulingUsingCSPAdaptiveJoint(SchedulingUsingCSPOnlineMultiObjWarmStart):
    """
    Incremental-first gate, but escalation is the FULL joint replan (the new job + every
    already-running job, exactly SchedulingUsingCSPOnline's own jobs_to_reschedule), not a
    single-job-only solve like SchedulingUsingCSPAdaptive. Matches this flow:

        new job arrives -> place with Incremental alone -> F1 (no extra cost)
                         -> F1 high enough? -> no: use that Incremental placement, done
                                             -> yes: pre-processing (freeze large / near-done /
                                                mid-transfer jobs among the OTHERS, via the same
                                                freeze_* config schedulingUsingJavaCSP already
                                                honors) -> joint online_biobj_warmstart replan,
                                                budgeted at adaptive_alpha * F1
                                                -> solution found: commit it (covers every job)
                                                -> no solution: fall back to Incremental for ONLY
                                                   the just-arrived job; every other already-
                                                   running job's plan is left exactly as it was
                                                   (Incremental never reconsiders them anyway, and
                                                   nothing forced them to change).

    Pre-processing matters here specifically because the escalation is a JOINT solve: with many
    already-running jobs also in the batch, freezing the ones unlikely to benefit from
    reconsideration (see freeze_large_jobs_threshold_mb / freeze_remaining_time_threshold /
    freeze_jobs_with_ongoing_transfer) keeps that joint solve's search space down. It's still an
    opt-in speed heuristic, not a correctness requirement -- if the frozen configuration happens
    to be infeasible (confirmed possible: multiple simultaneously-frozen jobs can jointly
    overconstrain the model even with no shared node between them), the "no solution -> fall back
    to Incremental for the new job" path above is what keeps this safe, not a guarantee that
    freezing itself never breaks feasibility.
    """
    java_main_class = 'MainIncremental'
    escalation_java_main_class = 'MainOnlineMultiObjWarmStart'
    adaptive_alpha = 0.2
    adaptive_max_budget_s = 1200
    # F1 must exceed this to bother escalating at all; None (the default) means "always try" --
    # budget already scales with F1, so a small job naturally gets a small (cheap) escalation
    # attempt rather than being blocked from one entirely.
    adaptive_f1_threshold = None
    # Budget for the internal Incremental calls (F1 probe + fallback), independent of
    # --solver-time-limit. None (the default) falls back to whatever solver_time_limit_s
    # currently is.
    incremental_time_limit_s = None
    def _flowTimeFromPlan(self, job, transfers_, works_):
        """Same as SchedulingUsingCSPAdaptive's: this job's own flow time (finish - arrival) from
        a plan's absolute times, or None if the job doesn't appear in the plan at all."""
        end_times = []
        for entries in transfers_.values():
            for t_job_id, t_node, t_start, t_end, t_dur in entries:
                if t_job_id == job.job_id:
                    end_times.append(t_end)
        for entries in works_.values():
            for w_job_id, w_node, w_task, w_start, w_end, w_dur in entries:
                if w_job_id == job.job_id:
                    end_times.append(w_end)
        if not end_times:
            return None
        return max(end_times) - job.arriving_time

    def _writeWarmStart(self, jobs_to_reschedule, replicas_locations, nodes_free_time):
        """Overrides SchedulingUsingCSPOnlineWarmStart._writeWarmStart(): that version's "brand-
        new job(s)" loop iterates self.waiting_jobs directly, which is correct THERE because its
        own schedulingNewJob() always puts every currently-waiting job into jobs_to_reschedule
        together. Here, only ONE job at a time is ever escalated (self.waiting_jobs[0]) while
        other, unrelated jobs can simultaneously sit in self.waiting_jobs still unprocessed --
        iterating self.waiting_jobs there would build placements keyed by a job_id job_index
        never learned about (KeyError). Deriving "new" from jobs_to_reschedule itself instead is
        correct in both cases and doesn't depend on what else happens to be waiting."""
        sorted_jobs = sorted(jobs_to_reschedule, key=lambda j: j.job_id)
        job_index = {job.job_id: idx for idx, job in enumerate(sorted_jobs)}
        now = self.env.now

        job_placements = []
        transfers_ws = []

        already_known_ids = {job.job_id for job in self.getRunningJobs()}
        for node_id in range(len(self.compute_nodes)):
            key = f'node_{node_id}'
            for work in self.works.get(key, []):
                w_job_id, w_node, w_task, w_start, w_end, w_dur = work
                if w_job_id in job_index and w_job_id in already_known_ids:
                    job_placements.append({
                        "job_index": job_index[w_job_id], "task_index": int(w_task),
                        "node": int(w_node), "start": int(round(w_start - now)),
                    })
            for transfer in self.transfers.get(key, []):
                t_job_id, t_node, t_start, t_end, t_dur = transfer
                if t_job_id in job_index and t_job_id in already_known_ids:
                    transfers_ws.append({
                        "job_index": job_index[t_job_id], "node": int(t_node),
                        "start": int(round(t_start - now)),
                    })

        new_jobs = [job for job in jobs_to_reschedule if job.job_id not in already_known_ids]
        orig_java_main_class = self.java_main_class
        # incremental_time_limit_s applies here too -- this throwaway call is itself an
        # Incremental solve, and without this it silently ran under whatever solver_time_limit_s
        # was set for the OUTER context instead (confirmed: with --solver-time-limit 600 this made
        # every escalating job pay up to an extra 600s just to build its own warm start).
        incremental_limit = self._config.get('incremental_time_limit_s', self.incremental_time_limit_s)
        orig_limit = self._config.get('solver_time_limit_s')
        try:
            if incremental_limit is not None:
                self._config['solver_time_limit_s'] = max(1, int(round(incremental_limit)))
            for job in new_jobs:
                self.java_main_class = 'MainIncremental'
                inc_transfers, inc_works, _ = schedulingUsingJavaCSP(
                    self, [job], replicas_locations, nodes_free_time, self.env.now)
                for node_id in range(len(self.compute_nodes)):
                    key = f'node_{node_id}'
                    for work in inc_works.get(key, []):
                        w_job_id, w_node, w_task, w_start, w_end, w_dur = work
                        if w_job_id == job.job_id:
                            job_placements.append({
                                "job_index": job_index[w_job_id], "task_index": int(w_task),
                                "node": int(w_node), "start": int(round(w_start - now)),
                            })
                    for transfer in inc_transfers.get(key, []):
                        t_job_id, t_node, t_start, t_end, t_dur = transfer
                        if t_job_id == job.job_id:
                            transfers_ws.append({
                                "job_index": job_index[t_job_id], "node": int(t_node),
                                "start": int(round(t_start - now)),
                            })
        finally:
            self.java_main_class = orig_java_main_class
            if incremental_limit is not None:
                if orig_limit is None:
                    self._config.pop('solver_time_limit_s', None)
                else:
                    self._config['solver_time_limit_s'] = orig_limit

        warm_start = {"job_placements": job_placements, "transfers": transfers_ws}
        with open(os.path.join(solver_model_dir(), "inputs", "warm_start.json"), "w") as f:
            json.dump(warm_start, f)

    # Opt-in: run the escalation as TWO concurrent solves (warm-started vs cold) and keep
    # whichever finds the better phase-1 optimum, instead of only ever warm-starting. See
    # _timedParallelEscalation's own docstring for why (a warm-started time-limited search can
    # converge to a MUCH worse optimum than a cold one on a bigger joint problem -- confirmed:
    # identical 15s phase-1 budget, warm-started result 2592 vs cold 1228 on the same 4-job
    # batch). None (the default): single warm-started solve only, identical to before this
    # existed.
    parallel_warm_cold_escalation = False

    # Labels for the concurrent escalation variants -- see _timedParallelEscalation. Four
    # variants, crossing "which jobs get frozen" (below-median / above-median / none) with
    # "warm-started or not" (only the no-freeze case tries both, since the earlier warm-vs-cold
    # investigation already found no meaningful difference between them: 13/14 controlled
    # comparisons landed on the identical optimum) -- see _estimateJobFlowTime /
    # _splitRunningJobsByMedianFlowTime for how the two groups are chosen each escalation.
    _PARALLEL_ESCALATION_LABELS = ("freeze_below_median", "freeze_above_median", "nofreeze", "warm_nofreeze")

    def _ensureParallelRunDirs(self):
        """Lazily creates one isolated utils/model tree per concurrent escalation variant (own
        inputs/outputs/bin -- javac recompiles into bin/ on every solve, so concurrent threads
        can't share one without racing each other's compile -- lib/src symlinked back to the
        canonical copy, which is read-only and safe to share). Cached on self after the first
        call. Base directory is wherever this process's own solves already resolve to
        (SIMULATOR_RUN_CWD if set, else the canonical simulator checkout)."""
        cached = getattr(self, '_parallel_run_cwds', None)
        if cached is not None:
            return cached
        base_run_cwd = os.environ.get("SIMULATOR_RUN_CWD", SIMULATOR_DIR)
        canonical_model_dir = os.path.join(base_run_cwd, "utils", "model") if base_run_cwd != SIMULATOR_DIR else MODEL_DIR
        run_cwds = {}
        for label in self._PARALLEL_ESCALATION_LABELS:
            run_cwd = os.path.join(base_run_cwd, f"parallel_{label}_escalation")
            model_dir = os.path.join(run_cwd, "utils", "model")
            for sub in ("inputs", "outputs", "bin"):
                os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
            for shared in ("lib", "src"):
                link_path = os.path.join(model_dir, shared)
                if not os.path.islink(link_path) and not os.path.exists(link_path):
                    os.symlink(os.path.join(canonical_model_dir, shared), link_path)
            run_cwds[label] = run_cwd
        self._parallel_run_cwds = run_cwds
        return run_cwds

    def _ensureEscalationPool(self):
        """Lazily creates ONE ThreadPoolExecutor for the whole simulation's lifetime, reused by
        every _timedParallelEscalation call, instead of a fresh pool per replan. A run of ~5
        escalations on Grid5000 (Linux) was observed to accumulate 65 live threads and then hang
        forever with one Python thread pegged at ~100% CPU and zero Java subprocesses left --
        i.e. stuck inside ThreadPoolExecutor's own bookkeeping after every variant had already
        returned, the same symptom the old per-call `pool.shutdown(wait=False)` was already known
        to trigger on macOS. Reusing a single long-lived pool (never shut down mid-run) removes
        the repeated create/destroy cycle that's the leading suspect for that bookkeeping getting
        corrupted, and incidentally also fixes the thread-count leak itself."""
        pool = getattr(self, '_escalation_pool', None)
        if pool is None:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self._PARALLEL_ESCALATION_LABELS))
            self._escalation_pool = pool
        return pool

    def _estimateJobFlowTime(self, job):
        """Iterative greedy list-scheduling estimate of this job's OWN total flow time, ported
        from replication-policies-simulator's own estimateFlowTime/checkUtility
        (utils/searchAlgo.py). Two things distinguish this from a naive "remaining work / number
        of replicas" average: (1) it uses job.nb_tasks (the FULL task count, not just
        NotStarted) recomputed from job.arriving_time across the CURRENT replica set -- i.e.
        "what would this job's flow time be if this exact replica set had been load-balanced
        since arrival", a consistent quantity across jobs regardless of how much of each has
        actually executed so far; (2) it accounts for each replica's own transfer-completion
        time and its node's own speed (compute_capacity), not an equal split. Job.replicas is
        populated by startTransfer() with real transfer_start_time/transfer_time per replica.
        Simulates greedily: repeatedly hand the next task to whichever replica becomes free
        earliest, which is the standard approximation for makespan-minimizing scheduling on
        unrelated machines and naturally balances finish times across replicas of different
        speed and staggered availability. Returns None if the job has no replicas yet (nothing
        to estimate from)."""
        if not job.replicas:
            return None
        task_duration = job.tasks[0].duration if job.tasks else 0
        next_free = {}
        for rep in job.replicas:
            available_at = (rep.transfer_start_time - job.arriving_time) + rep.transfer_time
            speed = task_duration * self.compute_nodes[rep.node_id].compute_capacity
            # A job can end up with more than one Replica entry for the same node across
            # replans (e.g. re-confirmed on a later escalation) -- keep only the latest, it
            # reflects the most current transfer, not a phantom duplicate capacity.
            next_free[rep.node_id] = [available_at, speed]
        for _ in range(job.nb_tasks):
            n = min(next_free, key=lambda node_id: next_free[node_id][0])
            next_free[n][0] += next_free[n][1]
        return max(state[0] for state in next_free.values())

    def _splitRunningJobsByMedianFlowTime(self, jobs):
        """(below, above): `jobs` split around the MEDIAN of their own _estimateJobFlowTime --
        jobs with no estimate yet (no replicas, e.g. still mid-first-transfer-decision) are
        excluded from both groups entirely (left unfrozen either way, since there's nothing to
        rank them by). Ties go to the below-median group (arbitrary but deterministic)."""
        estimates = [(job, self._estimateJobFlowTime(job)) for job in jobs]
        measurable = [(job, est) for job, est in estimates if est is not None]
        if not measurable:
            return [], []
        measurable.sort(key=lambda pair: pair[1])
        median = measurable[len(measurable) // 2][1]
        below = [job for job, est in measurable if est <= median]
        above = [job for job, est in measurable if est > median]
        return below, above

    def _timedParallelEscalation(self, jobs_to_reschedule, not_finished_jobs, replicas_locations, nodes_free_time, now, budget):
        """Runs the escalation as FOUR concurrent Java solves, crossing "which not-finished jobs
        get frozen" with "warm-started or not":
          - freeze_below_median: cold search, freeze whichever not-finished jobs have the SMALLER
            half of _estimateJobFlowTime (i.e. the jobs closest to done / with the least to gain
            from reconsideration -- see the whole freeze-criterion investigation this replaces).
          - freeze_above_median: cold search, freeze the LARGER half instead -- included as the
            direct comparison point, expected to reproduce the same regression the old
            large-dataset/ongoing-transfer freeze criteria caused (confining a job with lots of
            remaining work to a single node kills its own parallelism).
          - nofreeze: cold search, nothing frozen at all.
          - warm_nofreeze: warm-started search (escalation_java_main_class, seeded via
            _writeWarmStart), nothing frozen -- kept only to double check warm-vs-cold still
            doesn't matter once freezing is out of the picture (the earlier warm-vs-cold
            investigation already found 13/14 controlled comparisons identical).
        The new arriving job (jobs_to_reschedule[0]) is never frozen in any variant -- only
        not_finished_jobs are eligible, and only those with an existing replica (median split
        needs an estimate; see _splitRunningJobsByMedianFlowTime).

        Real OS-level parallelism: schedulingUsingJavaCSP blocks on a Java subprocess, which
        releases the GIL, so the threads' Java processes genuinely run side by side on separate
        cores, not one after the other. Each thread gets its own utils/model tree
        (_ensureParallelRunDirs) and its own shallow-copied master_node "view" (independent
        java_main_class/_config, everything else -- env, compute_nodes, jobs -- shared by
        reference and only ever READ during a solve) so no thread's bookkeeping races another's.
        Charges the REAL wall-clock time of the SLOWEST variant against simulated time (they ran
        concurrently, not back to back) -- callers must use `yield from`."""
        run_cwds = self._ensureParallelRunDirs()

        below_median_jobs, above_median_jobs = self._splitRunningJobsByMedianFlowTime(not_finished_jobs)
        below_ids = [j.job_id for j in below_median_jobs]
        above_ids = [j.job_id for j in above_median_jobs]
        print(f"### PARALLEL ESCALATION median split -- below (frozen in freeze_below_median): "
              f"{below_ids}, above (frozen in freeze_above_median): {above_ids} ###")

        def run_variant(label, java_main_class, write_warm_start, frozen_job_ids):
            proxy = copy.copy(self)
            proxy._config = dict(self._config)
            proxy.java_main_class = java_main_class
            proxy._config['solver_time_limit_s'] = max(1, int(round(budget)))
            # Bypasses the threshold-based freeze_large_jobs_threshold_mb/freeze_remaining_time_
            # threshold/freeze_jobs_with_ongoing_transfer criteria entirely: this experiment picks
            # the exact frozen set itself (median-based), not via a fixed magic threshold.
            proxy._config['frozen_job_ids_override'] = frozen_job_ids
            with run_cwd_override(run_cwds[label]):
                if write_warm_start:
                    proxy._writeWarmStart(jobs_to_reschedule, replicas_locations, nodes_free_time)
                start = time.time()
                transfers_, works_, deletions_ = schedulingUsingJavaCSP(
                    proxy, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
                elapsed = time.time() - start
            return transfers_, works_, deletions_, elapsed

        variant_specs = {
            "freeze_below_median": ("MainOnlineMultiObj", False, below_ids),
            "freeze_above_median": ("MainOnlineMultiObj", False, above_ids),
            "nofreeze": ("MainOnlineMultiObj", False, []),
            "warm_nofreeze": (self.escalation_java_main_class, True, []),
        }
        # Safety timeout, not just a nicety: with 3 concurrent variants, this was observed to
        # occasionally hang indefinitely on macOS (0 Java processes left running, one Python
        # thread pegged at 100% CPU, seemingly stuck inside ThreadPoolExecutor's own bookkeeping
        # after every variant's solve had already returned a value) -- root cause not fully
        # isolated (possibly a CPython/macOS thread-pool interaction after several rounds of
        # concurrent subprocess forking; unconfirmed whether Linux/Grid5000 is affected the same
        # way). Bounding each future's wait means a hang degrades to "this variant is skipped",
        # never "the whole simulation is stuck forever". pool.shutdown(wait=False): don't also
        # block on joining worker threads here, in case THAT is where a hang actually sits --
        # any lingering thread is harmless (it never touches shared state after returning).
        result_timeout_s = budget + 90
        pool = self._ensureEscalationPool()
        print(f"### PARALLEL ESCALATION diag: submitting at {time.strftime('%H:%M:%S')}, "
              f"active_count={threading.active_count()} ###")
        futures = {
            label: pool.submit(run_variant, label, java_main_class, write_warm_start, frozen_job_ids)
            for label, (java_main_class, write_warm_start, frozen_job_ids) in variant_specs.items()
        }
        results = {}
        for label, future in futures.items():
            try:
                results[label] = future.result(timeout=result_timeout_s)
                print(f"### PARALLEL ESCALATION diag: '{label}' returned at {time.strftime('%H:%M:%S')} ###")
            except Exception as e:
                print(f"### WARNING: parallel escalation variant '{label}' failed/timed out ({e}) -- treated as no solution ###")
                results[label] = ({}, {}, {}, result_timeout_s)
        print(f"### PARALLEL ESCALATION diag: all gathered at {time.strftime('%H:%M:%S')}, "
              f"active_count={threading.active_count()} ###")

        def batch_quality(works_):
            """(max flow time, total energy) for this candidate -- lower is better on both,
            energy only breaks ties on max flow time. None (no solution) sorts last."""
            if not works_:
                return None
            finish = {}
            for entries in works_.values():
                for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                    finish[job_id] = end_abs if job_id not in finish else max(finish[job_id], end_abs)
            arrival_by_id = {j.job_id: j.arriving_time for j in jobs_to_reschedule}
            flows = [finish[jid] - arrival_by_id[jid] for jid in finish if jid in arrival_by_id]
            if not flows:
                return None
            return (max(flows), 0.0)

        qualities = {label: batch_quality(works_) for label, (_, works_, _, _) in results.items()}
        print("### PARALLEL ESCALATION: " + ", ".join(
            f"{label}={qualities[label]} ({results[label][3]:.3f}s)" for label in variant_specs) + " ###")

        ranked = [label for label in variant_specs if qualities[label] is not None]
        ranked.sort(key=lambda label: qualities[label])
        if ranked:
            best = ranked[0]
            transfers_, works_, deletions_, _ = results[best]
        else:
            transfers_, works_, deletions_ = {}, {}, {}

        charge_thinking_time = self._config.get('charge_thinking_time', self.charge_thinking_time)
        if charge_thinking_time:
            yield self.env.timeout(max(elapsed for (_, _, _, elapsed) in results.values()))

        return transfers_, works_, deletions_

    def _placeSingleJobIncremental(self, job):
        """Throwaway (or final-fallback) single-job Incremental solve, using Incremental's own
        node-availability semantics (nodesFreeTimeIncremental accounts for queued backlog, unlike
        plain nodesFreeTime). Budgeted at incremental_time_limit_s (config or class attr) when
        set, independently of --solver-time-limit -- Incremental's own placement is meant to be
        cheap/fast, so it shouldn't have to share the (often much larger) budget used elsewhere
        for the joint bi-objectif escalation or for a plain "incremental" approach run standalone.
        Generator (charges this solve's own real wall-clock time via _timedSchedulingUsingJavaCSP,
        see charge_thinking_time) -- callers must use `yield from`."""
        nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(
            self, self.ongoing_transfers, self.ongoing_works)
        orig_java_main_class = self.java_main_class
        incremental_limit = self._config.get('incremental_time_limit_s', self.incremental_time_limit_s)
        orig_limit = self._config.get('solver_time_limit_s')
        try:
            self.java_main_class = 'MainIncremental'
            if incremental_limit is not None:
                self._config['solver_time_limit_s'] = max(1, int(round(incremental_limit)))
            transfers_, works_, deletions_ = yield from self._timedSchedulingUsingJavaCSP(
                [job], self.replicas_locations, nodes_free_time, self.env.now)
        finally:
            self.java_main_class = orig_java_main_class
            if incremental_limit is not None:
                if orig_limit is None:
                    self._config.pop('solver_time_limit_s', None)
                else:
                    self._config['solver_time_limit_s'] = orig_limit
        flow_time = self._flowTimeFromPlan(job, transfers_, works_)
        return transfers_, works_, deletions_, flow_time

    def _commitSingleJobPlan(self, transfers_, works_, deletions_):
        """Append-only merge (mirrors SchedulingUsingCSPIncremental.schedulingNewJob): only this
        one job's entries are added, nothing already committed for any other job is touched."""
        for node in range(len(self.compute_nodes)):
            key = "node_" + str(node)
            if key in transfers_ and len(transfers_[key]) > 0:
                for transfer in transfers_[key]:
                    self.transfers[key].append(transfer)
            if key in works_ and len(works_[key]) > 0:
                for work in works_[key]:
                    self.works[key].append(work)
            if key in deletions_ and len(deletions_[key]) > 0:
                for deletion in deletions_[key]:
                    self.deletions[key].append(deletion)

    def _commitJointPlan(self, transfers_, works_, deletions_):
        """Full-replan merge (mirrors SchedulingUsingCSPOnlineWarmStart.schedulingNewJob): every
        node's list is reset and rebuilt from this joint plan, since it re-decided everything."""
        for node in range(len(self.compute_nodes)):
            key = "node_" + str(node)
            self.transfers[key] = []
            self.works[key] = []
            self.deletions[key] = []
            if key in transfers_ and len(transfers_[key]) > 0:
                ongoing = self.ongoing_transfers.get(key)
                for transfer in transfers_[key]:
                    transfer_job_id = transfer[0]
                    already_present = transfer_job_id in self.replicas_locations and node in self.replicas_locations[transfer_job_id]
                    already_in_flight = ongoing is not None and ongoing[0] == transfer_job_id
                    if already_present or already_in_flight:
                        continue
                    self.transfers[key].append(transfer)
            # Bug found 2026-09-27: this used to be nested inside the `if key in transfers_`
            # block above, so a node with real task placements but NO new transfer this round
            # (the normal case: the job's data is already resident there from an earlier round,
            # e.g. a frozen job confined to its existing replicas) silently lost its works
            # entries every time -- the affected job's remaining tasks then stayed "NotStarted"
            # forever, finished_jobs never reached total_nb_jobs, and the simulation spun in
            # place (schedulingNewJob/checkOnJobs polling every 0.1 with nothing left to do) for
            # the rest of the run. Present since _commitJointPlan's introduction (e1de3a4), just
            # never triggered until frozen-job escalation made "no new transfer, real work exists"
            # common. Works/transfers are independent outputs of the same solve -- neither should
            # gate the other.
            if key in works_ and len(works_[key]) > 0:
                for work in works_[key]:
                    self.works[key].append(work)
            if key in deletions_ and len(deletions_[key]) > 0:
                for deletion in deletions_[key]:
                    self.deletions[key].append(deletion)

    def schedulingNewJob(self):

        poll_count = 0
        while True:
            yield self.env.timeout(0.1)
            poll_count += 1
            if poll_count % 2000 == 0:
                # Diagnostic added to isolate a 2026-09-27 Grid5000 hang: this loop polling
                # forever with no escalation in flight would show up here (large poll_count,
                # waiting_jobs empty, _allJobsCompleted() stuck False) as opposed to the hang
                # being inside _timedParallelEscalation's ThreadPoolExecutor bookkeeping instead.
                print(f"### DIAG schedulingNewJob poll #{poll_count} at sim_time={self.env.now:.2f}: "
                      f"waiting_jobs={len(self.waiting_jobs)}, finished_jobs={self.finished_jobs}/{self._config['total_nb_jobs']}, "
                      f"node_queues={[len(cn.queue.items) for cn in self.compute_nodes]} ###")

            if len(self.waiting_jobs) >= 1:

                job = self.waiting_jobs[0]

                logger.debug("[%s] Master: probing Incremental for job %s (adaptive-joint)", self.env.now, job.job_id)
                inc_transfers, inc_works, inc_deletions, f1 = yield from self._placeSingleJobIncremental(job)

                if f1 is None:
                    logger.warning("[%s] Master: no CSP solution found for job %s (adaptive-joint/incremental probe), will retry", self.env.now, job.job_id)
                    if self._allJobsCompleted():
                        break
                    continue

                alpha = self._config.get('adaptive_alpha', self.adaptive_alpha)
                threshold = self._config.get('adaptive_f1_threshold', self.adaptive_f1_threshold)
                should_escalate = threshold is None or f1 > threshold

                if should_escalate:
                    max_budget = self._config.get('adaptive_max_budget_s', self.adaptive_max_budget_s)
                    budget = min(alpha * f1, max_budget)
                    logger.debug("[%s] Master: job %s F1=%.3f -> joint escalation budget=%.3f",
                                 self.env.now, job.job_id, f1, budget)

                    nodes_free_time = self.nodesFreeTime(self.ongoing_transfers, self.ongoing_works)
                    replicas_locations = self.replicas_locations
                    jobs_to_reschedule = [job] + self.getRunningJobs()

                    parallel_escalation = self._config.get('parallel_warm_cold_escalation', self.parallel_warm_cold_escalation)
                    if parallel_escalation:
                        # Runs 4 candidate escalations CONCURRENTLY (freeze the below-median
                        # group / freeze the above-median group / no freeze / no freeze but
                        # warm-started) and keeps the best -- see _timedParallelEscalation's own
                        # docstring. Handles its own java_main_class/_config/warm-start/freeze/
                        # timing via per-thread proxies, so nothing here needs mutating self.
                        transfers_, works_, deletions_ = yield from self._timedParallelEscalation(
                            jobs_to_reschedule, self.getRunningJobs(), replicas_locations, nodes_free_time, self.env.now, budget)
                    else:
                        self._writeWarmStart(jobs_to_reschedule, replicas_locations, nodes_free_time)
                        orig_limit = self._config.get('solver_time_limit_s')
                        orig_java_main_class = self.java_main_class
                        try:
                            self._config['solver_time_limit_s'] = max(1, int(round(budget)))
                            # Without this, the joint solve silently runs under the class-level
                            # default (MainIncremental, kept for the cheap F1 probe) instead of the
                            # actual bi-objectif+warmstart escalation -- and since MainIncremental
                            # never stops early once it can't quickly prove optimal, it then just
                            # burns the entire budget doing the wrong solve (confirmed: a real run got
                            # stuck for 5+ minutes on a single-job MainIncremental call carrying a
                            # 619s budget meant for the joint replan).
                            self.java_main_class = self.escalation_java_main_class
                            # _timedSchedulingUsingJavaCSP charges this solve's REAL wall-clock time
                            # (not the budget estimate above, which only bounds the solver's time
                            # limit) against simulated time -- see charge_thinking_time. Replaced an
                            # earlier version that pre-emptively charged `budget` itself (a fixed
                            # estimate, not what the solve actually took) before even running the
                            # solve; this way every approach (online/online_biobj/incremental/hybrid)
                            # is charged the same way, for its own actual solve, not an estimate.
                            transfers_, works_, deletions_ = yield from self._timedSchedulingUsingJavaCSP(
                                jobs_to_reschedule, replicas_locations, nodes_free_time, self.env.now)
                        finally:
                            self.java_main_class = orig_java_main_class
                            if orig_limit is None:
                                self._config.pop('solver_time_limit_s', None)
                            else:
                                self._config['solver_time_limit_s'] = orig_limit

                    if len(transfers_.keys()) > 0 and len(works_.keys()) > 0:
                        # Quality gate: the escalation is a full joint replan under a TIGHT budget
                        # (and, in the parallel case, split across freeze variants) -- nothing
                        # about it guarantees the new job itself comes out ahead of what the cheap
                        # Incremental probe already found (F1). Compare the two on THIS job's own
                        # flow time (the only metric directly comparable between a single-job
                        # placement and a joint replan) and keep whichever is actually better,
                        # instead of accepting the escalation unconditionally just because it
                        # found *a* solution. inc_transfers/inc_works/inc_deletions are the exact
                        # placement the F1 probe already computed above -- reused here instead of
                        # re-solving.
                        escalation_flow = self._flowTimeFromPlan(job, transfers_, works_)
                        if escalation_flow is not None and escalation_flow > f1:
                            logger.debug("[%s] Master: job %s joint escalation (flow=%.3f) worse than "
                                         "F1 (%.3f) -- keeping Incremental's placement instead",
                                         self.env.now, job.job_id, escalation_flow, f1)
                            self._commitSingleJobPlan(inc_transfers, inc_works, inc_deletions)
                        else:
                            logger.debug("[%s] Master: job %s joint escalation accepted (%s job(s) replanned)",
                                         self.env.now, job.job_id, len(jobs_to_reschedule))
                            self._commitJointPlan(transfers_, works_, deletions_)
                        self.waiting_jobs.pop(0)
                    else:
                        logger.warning("[%s] Master: joint escalation found no solution for job %s, "
                                       "keeping Incremental's placement (already computed above)",
                                       self.env.now, job.job_id)
                        self._commitSingleJobPlan(inc_transfers, inc_works, inc_deletions)
                        self.waiting_jobs.pop(0)
                else:
                    logger.debug("[%s] Master: job %s F1=%.3f below threshold, using Incremental placement as-is",
                                 self.env.now, job.job_id, f1)
                    self._commitSingleJobPlan(inc_transfers, inc_works, inc_deletions)
                    self.waiting_jobs.pop(0)

            if self._allJobsCompleted():
                break


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

            self.tracker.log_transfer(
                job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id,
                receiver_energy_consumption=compute_node.energy_consumption,
                sender_energy_consumption=self._config.get('master_energy_consumption', 0.0),
                network_energy=self._config.get('network_energy_per_transfer', 0.0),
            )

            if compute_node.node_id in self.actual_transfers.keys(): del self.actual_transfers[self.compute_nodes[compute_node.node_id].node_id]

