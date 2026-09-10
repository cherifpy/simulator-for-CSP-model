package main;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.chocosolver.solver.Model;
import org.chocosolver.solver.Settings;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.search.loop.monitors.IMonitorSolution;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainMedian;
import org.chocosolver.solver.search.strategy.selectors.variables.RandomVar;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.util.objects.setDataStructures.iterable.IntIterableRangeSet;
import org.chocosolver.util.tools.ArrayUtils;
import java.util.Arrays;
import org.chocosolver.solver.variables.Task;
import java.util.Comparator;
import java.util.stream.IntStream;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Online/dynamic adaptation of the "three-step" bag-of-tasks scheduling approach from the
 * (offline, single-shot) flowtime-scheduler-main project: Dataset Splitting -> per-node Transfer
 * Scheduling -> per-node Task Allocation, iterated with cuts (see ThreeStepScheduling2 there).
 * <p>
 * This is a NEW, standalone entry point -- it does not modify MainOnline.java/MainIncremental.java
 * at all. It reads/writes the exact same file contract as MainOnline.java (same
 * utils/model/inputs/*.json and utils/model/outputs/*.csv), so it plugs into the existing
 * Python-side simulator (utils/modelCSP.py) unchanged: just point a scheduler's
 * `java_main_class` at "MainOnlineThreeStep".
 * <p>
 * Two things needed changing to make the originally-offline three-step algorithm work in our
 * dynamic, per-solve setting (per the task): 1) every node has its own "starting time" (when it
 * actually becomes free, from ongoing transfers/work -- computed on the Python side) instead of
 * being free from t=0; and 2) transfer duration must account for data already resident on a node
 * from a previous solve (near-free reuse) instead of always paying size/bandwidth. Both are
 * threaded through DatasetAssignmentProblemNodeJ/TaskAllocationProblemNodeJ (see their javadoc).
 * <p>
 * Storage capacity (neither the original 2-step nor 3-step approach modeled it) is added on top,
 * mirroring MainOnline.java's design: Step 1 forbids assigning any task of a dataset to a node too
 * small to ever hold it at all; Step 2 pins an already-resident dataset's (near-instant) "transfer"
 * window to exactly nodeStartingTime instead of leaving it a free variable a solver could shrink to
 * hide real occupancy; Step 3 -- the only step that actually knows real task-completion times --
 * posts the per-node cumulative(storage) constraint over [transferStart, last-task-end] and, if the
 * capacity is exceeded, naturally falls into the existing "no solution -> BadAssignements hard cut"
 * feedback path, so Step 1/2 retry with a different plan next iteration. Abandon (keep-vs-delete)
 * decisions are read off the final chosen plan and written to deletions.csv exactly like
 * MainOnline.java's.
 * <p>
 * Known scalability limitation (observed on inst-10J-10N under real per-node storage capacities):
 * Step 1's objective only minimizes transfer time, with no notion of a node's compute speed (that's
 * Step 3's dimension entirely, invisible to Step 1) or of storage pressure (unknown until Step 3
 * rejects a combination) -- so it systematically piles data onto whichever nodes have the highest
 * bandwidth, even when those same nodes are slow to compute and/or already storage-constrained by
 * previously-resident (possibly over-replicated) datasets. Once several concurrent jobs need
 * placement, this bias can make Step 1/2/3's cutting-plane loop need far more outer iterations to
 * escape a bad pattern than the batch before it needed -- e.g. one batch cleared in 46 iterations
 * within a 60s budget, while the very next (larger) batch still hadn't cleared in 60s (~3 iterations
 * only, each outer iteration now costing more per node). Retrying at the SAME solver_time_limit_s
 * does not help escape this on its own -- every retry is a fresh JVM/solve starting from an empty
 * cutting-plane state, not a continuation of a previous attempt's search -- so only raising
 * solver_time_limit_s itself buys more iterations, and the number needed appears to grow with
 * concurrent job count rather than staying flat. Fixing this properly would mean giving Step 1 some
 * awareness of compute speed and/or storage pressure (e.g. re-enabling/tuning the currently-disabled
 * `alpha` parameter, which exists for exactly this transfer-vs-compute tradeoff) -- not attempted
 * here; flagged as a known limitation instead.
 */
public class MainOnlineThreeStep {

    private static final String SIMULATOR_DIR = System.getProperty("user.dir");
    private static final String MODEL_INPUTS_DIR = SIMULATOR_DIR + "/utils/model/inputs";
    private static final String MODEL_OUTPUTS_DIR = SIMULATOR_DIR + "/utils/model/outputs";

    public static class Job {
        public int job_id;
        public int datasetSize;
        public int nbTasks;
        public int taskDuration;
        public int timelasped;
        public double job_arriving_time;
    }

    public static class NodeConfig {
        public int bandwidth;
        public double computationNodes;
        public double freeTime;
        public int storageCapacity;
    }

    public static String readFile(String path) throws Exception {
        return new String(Files.readAllBytes(Paths.get(path)));
    }

    public static void writeTextFile(String path, String content) throws Exception {
        Files.write(Paths.get(path), content.getBytes());
    }

    public static void main(String[] args) throws Exception {

        // ---- Load jobs JSON ----
        JSONArray jobsArray = new JSONArray(readFile(MODEL_INPUTS_DIR + "/jobs.json"));
        List<Job> jobs = new ArrayList<>();
        for (int i = 0; i < jobsArray.length(); i++) {
            JSONObject j = jobsArray.getJSONObject(i);
            Job job = new Job();
            job.datasetSize = j.getInt("dataset_size");
            job.nbTasks = j.getInt("nb_tasks");
            job.taskDuration = j.getInt("task_duration");
            job.job_id = j.getInt("job_id");
            job.timelasped = j.getInt("timelasped");
            job.job_arriving_time = j.getFloat("job_arriving_time");
            jobs.add(job);
        }

        // ---- Load nodes JSON ----
        JSONArray nodesArray = new JSONArray(readFile(MODEL_INPUTS_DIR + "/nodes.json"));
        if (nodesArray.length() == 0) {
            throw new Exception("Error: missing informations");
        }
        List<NodeConfig> nodes = new ArrayList<>();
        for (int i = 0; i < nodesArray.length(); i++) {
            JSONObject j = nodesArray.getJSONObject(i);
            NodeConfig node = new NodeConfig();
            node.computationNodes = j.getFloat("compute_capacity");
            node.bandwidth = j.getInt("bandwidth");
            node.freeTime = j.getInt("free_time");
            node.storageCapacity = j.has("storage_capacity") ? j.getInt("storage_capacity") : Integer.MAX_VALUE / 2;
            nodes.add(node);
        }

        // ---- Load replicas_locations JSON (already-resident data per dataset) ----
        JSONArray replicasJson = new JSONArray(readFile(MODEL_INPUTS_DIR + "/replicas_locations.json"));
        int[][] replicas_location = new int[replicasJson.length()][];
        for (int i = 0; i < replicasJson.length(); i++) {
            JSONArray row = replicasJson.getJSONArray(i);
            replicas_location[i] = new int[row.length()];
            for (int k = 0; k < row.length(); k++) {
                replicas_location[i][k] = row.getInt(k);
            }
        }

        // ---- Solver time budget (wall-clock, seconds) ----
        int solverTimeLimitS = 30;
        try {
            solverTimeLimitS = Integer.parseInt(readFile(MODEL_INPUTS_DIR + "/solver_time_limit.txt").trim());
        } catch (Exception e) {
            // keep default
        }
        System.out.println("Solver time limit: " + solverTimeLimitS + "s");

        int nb_nodes = nodes.size();
        int nb_data = jobs.size();
        int[] data_sizes = new int[nb_data];
        int[][] works = new int[nb_data][];
        int[] flowTimeOffset = new int[nb_data]; // pass -timelasped so flow = jobEnd + timelasped
        int[] bandwidths = new int[nb_nodes];
        double[] cpus = new double[nb_nodes];
        int[] nodeStartingTimes = new int[nb_nodes]; // per-node earliest time a new decision can start
        int[] storage_capacity = new int[nb_nodes];

        for (int j = 0; j < nb_nodes; j++) {
            NodeConfig node = nodes.get(j);
            bandwidths[j] = node.bandwidth;
            cpus[j] = node.computationNodes;
            // Same margin as MainOnline.java's nodeStartingTimes[j]: the raw free-time estimate
            // from Python is a float, the model is integer -- ceil + 1 avoids ever under-shooting
            // (starting a decision fractionally before the node is really free).
            nodeStartingTimes[j] = (int) Math.ceil(node.freeTime) + 1;
            storage_capacity[j] = node.storageCapacity;
        }

        int i = 0;
        for (Job job : jobs) {
            data_sizes[i] = job.datasetSize;
            int[] taskDurations = new int[job.nbTasks];
            java.util.Arrays.fill(taskDurations, job.taskDuration);
            works[i] = taskDurations;
            flowTimeOffset[i] = -job.timelasped;
            i++;
        }

        // Dummy per-dataset "arrival" array for Step 1's (informational-only, never used in its
        // actual solving logic) debug printing -- kept for signature compatibility.
        int[] step1DebugOnlyArrivals = new int[nb_data];

        // ----- Iterative 3-step loop (Dataset Splitting -> per-node Transfer -> per-node Task
        // Allocation), bounded by a wall-clock deadline instead of running to full convergence
        // like the original offline algorithm (which has no time budget at all). -----
        long deadline = System.currentTimeMillis() + (long) solverTimeLimitS * 1000L;
        double alpha = 0.;

        List<Commons.Assignment> assignments = new ArrayList<>();
        List<Commons.Cut> cuts = new ArrayList<>();
        int bestMaxFlowTime = Integer.MAX_VALUE;
        List<Commons.Plan> bestPlan = new ArrayList<>();
        List<Commons.ScheduledAssignment> bestScheduledAssignments = new ArrayList<>();
        boolean foundAny = false;
        int iteration = 0;

        do {
            iteration++;
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) break;
            // Step 1 runs once per outer iteration: cap it well below the full budget so
            // Step 2/3 (run once per node, below) always get a share of the remaining time too.
            final List<Commons.Assignment> currentAssignments = DatasetSplittingProblem.runScheduler(
                    nb_data, data_sizes, works, step1DebugOnlyArrivals, nb_nodes, bandwidths, cpus, alpha,
                    assignments, cuts, false, iteration, Math.min(remaining, 2000L), storage_capacity);

            if (currentAssignments.isEmpty()) {
                System.out.println("Step 1: no valid assignment found, stopping.");
                break;
            }

            List<Commons.ScheduledAssignment> currentScheduledAssignments = new ArrayList<>();
            List<Commons.Plan> currentPlans = new ArrayList<>();
            List<Commons.Cut> hardCuts = new ArrayList<>();
            List<Commons.Cut> softCuts = new ArrayList<>();
            int maxFlowTime = 0;
            int sumFlowTime = 0;
            boolean ranOutOfTimeMidIteration = false;

            for (int j = 0; j < nb_nodes; j++) {
                remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) {
                    ranOutOfTimeMidIteration = true;
                    break;
                }
                final int fj = j;
                List<Commons.Assignment> assignmentsOfJ = currentAssignments.stream().filter(a -> a.n() == fj).toList();
                if (assignmentsOfJ.isEmpty()) continue;

                List<Commons.ScheduledAssignment> scheduledJ = DatasetAssignmentProblemNodeJ.runScheduler(
                        j, nb_data, data_sizes, nodeStartingTimes[j], works, nb_nodes, bandwidths,
                        replicas_location, assignmentsOfJ, cuts, false, Math.min(remaining, 1000L));
                if (scheduledJ.isEmpty()) continue;
                currentScheduledAssignments.addAll(scheduledJ);

                remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) {
                    ranOutOfTimeMidIteration = true;
                    break;
                }
                Commons.Result r = TaskAllocationProblemNodeJ.runScheduler(
                        j, nb_data, works, cpus, flowTimeOffset, nodeStartingTimes[j],
                        scheduledJ, bestMaxFlowTime, false, Math.min(remaining, 1000L),
                        data_sizes, storage_capacity[j]);

                if (r.maxFlow() > -1) {
                    maxFlowTime = Math.max(maxFlowTime, r.maxFlow());
                    sumFlowTime += r.sumFlow();
                    currentPlans.addAll(r.plans());
                    softCuts.addAll(r.cuts());
                } else {
                    hardCuts.addAll(r.cuts());
                }
            }

            // A partial iteration (cut short mid-way through the per-node loop by the deadline)
            // only solved a subset of nodes -- its maxFlowTime/sumFlowTime and cuts are not
            // comparable to a full iteration's, so discard it rather than risk recording a
            // bogus "best" or a cut derived from incomplete information.
            if (ranOutOfTimeMidIteration) {
                break;
            }

            if (!hardCuts.isEmpty()) {
                cuts.addAll(hardCuts);
            } else {
                Optional<Commons.BadAssignements> worstSoftCut = softCuts.stream()
                        .filter(c -> c instanceof Commons.BadAssignements)
                        .map(c -> (Commons.BadAssignements) c)
                        .max(Comparator.comparingInt(Commons.BadAssignements::ect));
                worstSoftCut.ifPresent(cuts::add);

                if (maxFlowTime < bestMaxFlowTime) {
                    bestMaxFlowTime = maxFlowTime;
                    bestPlan = new ArrayList<>(currentPlans);
                    bestScheduledAssignments = new ArrayList<>(currentScheduledAssignments);
                    foundAny = true;
                    System.out.println("Iteration " + iteration + ": new best maxFlowTime=" + maxFlowTime + " sum=" + sumFlowTime);
                }
            }
            assignments = currentAssignments;
        } while (System.currentTimeMillis() < deadline);

        System.out.println("Best maxFlowTime found = " + bestMaxFlowTime + " after " + iteration + " iteration(s)");

        // ----- Write output CSVs, same format as MainOnline.java's outputs -----
        if (!foundAny) {
            System.out.println("No solution found");
            writeTextFile(MODEL_OUTPUTS_DIR + "/transfers.csv", "job_index,start_time,end_time,node_index\n");
            writeTextFile(MODEL_OUTPUTS_DIR + "/works.csv", "task_index,job_index,start_time,end_time,node_index\n");
            writeTextFile(MODEL_OUTPUTS_DIR + "/deletions.csv", "job_index,node_index,deletion_time\n");
            return;
        }

        try (FileWriter w = new FileWriter(MODEL_OUTPUTS_DIR + "/transfers.csv")) {
            w.write("job_index,start_time,end_time,node_index\n");
            for (Commons.ScheduledAssignment sa : bestScheduledAssignments) {
                int start = sa.e() - sa.td();
                w.write(sa.a().d() + "," + start + "," + sa.e() + "," + sa.a().n() + "\n");
            }
        }

        try (FileWriter w = new FileWriter(MODEL_OUTPUTS_DIR + "/works.csv")) {
            w.write("task_index,job_index,start_time,end_time,node_index\n");
            for (Commons.Plan p : bestPlan) {
                w.write(p.k() + "," + p.d() + "," + p.s() + "," + p.e() + "," + p.n() + "\n");
            }
        }

        // Abandon decisions: a dataset already resident on a node (from a previous solve) that
        // this plan does NOT keep/use there anymore -- mirrors MainOnline.java's exact condition
        // (transferHeights[j][i]==0 while replicas_location[i] already contains j), just read off
        // the final chosen plan instead of a single monolithic model's decision variables.
        Set<Long> chosenPairs = new HashSet<>();
        for (Commons.ScheduledAssignment sa : bestScheduledAssignments) {
            chosenPairs.add((long) sa.a().d() * nb_nodes + sa.a().n());
        }
        try (FileWriter w = new FileWriter(MODEL_OUTPUTS_DIR + "/deletions.csv")) {
            w.write("job_index,node_index,deletion_time\n");
            for (int di = 0; di < nb_data; di++) {
                for (int j : replicas_location[di]) {
                    if (!chosenPairs.contains((long) di * nb_nodes + j)) {
                        w.write(di + "," + j + "," + nodeStartingTimes[j] + "\n");
                    }
                }
            }
        }
    }


    // ===== Nested: three-step model classes (from flowtime-scheduler-main/threesteps, adapted) =====

    /**
     * Utility class containing common data structures and helper methods for the three-step scheduling algorithm.
     * This class provides the foundational types used throughout the scheduling process, including assignments,
     * scheduled assignments, plans, and various cut types used in the constraint programming approach.
     */
    public static class Commons {

        public static boolean iTransferFasterThanJ(int bandwidth_i, int bandwidth_j){
            return bandwidth_i > bandwidth_j;
        }

        public static boolean iCalculateFasterThanJ(double cpu_i, double cpu_j){
            return cpu_i < cpu_j;
        }

        /**
         * Generalizes a set of hard cuts to apply constraints to similar datasets and nodes.
         * For each BadFactor cut, this method creates new cuts for datasets and nodes with similar characteristics
         * (size, number of tasks, bandwidth, CPU). This helps in propagating constraints to related parts of the problem.
         *
         * @param hardCuts   the list of hard cuts to generalize
         * @param nb_nodes   the total number of nodes
         * @param nb_data    the total number of datasets
         * @param data_sizes array containing the size of each dataset
         * @param works      2D array where works[i][j] represents the work amount for task j of dataset i
         * @param bandwidths array containing the bandwidth of each node
         * @param cpus       array containing the CPU power of each node
         * @return a collection of generalized cuts
         */
        protected static Collection<? extends Cut> generalize(List<Cut> hardCuts,
                                                              int nb_nodes, int nb_data,
                                                              int[] data_sizes, int[][] works,
                                                              int[] bandwidths, double[] cpus) {
            List<Cut> gCuts = new ArrayList<>();
            for (Cut c : hardCuts) {
                if (c instanceof BadAssignements(List<ScheduledAssignment> as, int ect)) {
                    // (logging removed)
                    int n = as.getFirst().a.n;
                    int bandwidth = bandwidths[n];
                    double cpu = cpus[n];
                    for (int j = 0; j < nb_nodes; j++) {
                        if (j == n) continue;
                        if (iTransferFasterThanJ(bandwidth,bandwidths[j])
                                && iCalculateFasterThanJ(cpu, cpus[j])) {
                            List<ScheduledAssignment> as2 = new ArrayList<>();
                            for (ScheduledAssignment a : as) {
                                assert a.a.n == n;
                                Assignment a2 = new Assignment(a.a.d, j, a.a.k);
                                ScheduledAssignment sa2 = new ScheduledAssignment(a2, a.e, a.td);
                                as2.add(sa2);
                            }
                            gCuts.add(new BadAssignements(as2, ect));
                            // (logging removed)
                        }
                    }
                }
            }
            return gCuts;
        }

        /**
         * Generates a Mermaid Gantt chart representation of the current scheduling solution.
         * This method prints to standard output a Mermaid-compatible Gantt chart showing the distribution
         * of datasets across nodes and their scheduled tasks.
         *
         * @param nb_nodes                    the total number of nodes
         * @param currentScheduledAssignments the list of scheduled assignments to visualize
         * @param plans                       the list of plans (scheduled tasks) to include in the chart
         * @param bestMaxFlowTime             the best maximum flow time found, used as the chart title
         */
        static void mermaidIt(int nb_nodes, List<ScheduledAssignment> currentScheduledAssignments, List<Plan> plans, int bestMaxFlowTime) {
            System.out.printf("""
                    gantt
                    title Meilleure solution trouvée avec nouvelle version (%d)
                    dateFormat X
                    axisFormat %%s
                    """, bestMaxFlowTime);
            for (int j = 0; j < nb_nodes; j++) {
                System.out.printf("\tsection Node %d\n", j);
                final int fJ = j;
                List<ScheduledAssignment> as = currentScheduledAssignments.stream().filter(a -> a.a().n() == fJ).toList();
                for (ScheduledAssignment a : as) {
                    System.out.printf("\t\tData %d : done, %d, %d\n", a.a().d(), a.e() - a.td(), a.e());
                    List<Plan> ps = plans.stream().filter(p -> p.d() == a.a().d() && p.n() == fJ).toList();
                    for (Plan p : ps) {
                        System.out.printf("\t\tJob %d-%d : active, %d, %d\n", p.d(), p.k(), p.s(), p.e());
                    }
                }
            }
        }

        /**
         * Interface representing a cut in the constraint programming model.
         * Cuts are used to eliminate infeasible solutions and guide the search process
         * by adding constraints that prevent previously found invalid configurations.
         */
        public interface Cut {
        }

        /**
         * Record representing the result of a scheduling operation.
         * Contains the computed plans, any cuts generated during the process,
         * and metrics about the flow time (maximum and sum).
         *
         * @param plans   the list of scheduled tasks (plans)
         * @param cuts    the list of cuts generated during scheduling
         * @param maxFlow the maximum flow time across all tasks
         * @param sumFlow the sum of all flow times
         */
        public record Result(List<Plan> plans, List<Cut> cuts, int maxFlow, int sumFlow) {
        }

        /**
         * Record representing an assignment of tasks to a node.
         *
         * @param d the dataset index
         * @param n the node index where the dataset tasks are assigned
         * @param k the number of tasks from dataset d assigned to node n
         */
        public record Assignment(int d, int n, int k) {
        }

        /**
         * Record representing a scheduled assignment with timing information.
         * Extends Assignment with end time and transfer duration.
         *
         * @param a  the base assignment (dataset, node, number of tasks)
         * @param e  the end time of the transfer
         * @param td the transfer duration
         */
        public record ScheduledAssignment(Assignment a, int e, int td) {
        }

        /**
         * Record representing a scheduled task plan.
         * Defines when and where a specific task from a dataset will be executed.
         *
         * @param d the dataset index
         * @param n the node index where the task will be executed
         * @param k the task index within the dataset
         * @param s the start time of the task
         * @param e the end time of the task
         */
        public record Plan(int d, int n, int k, int s, int e) {
        }

        /**
         * Cut type indicating bad assignments that violate timing constraints.
         * This cut is generated when scheduled assignments would result in execution times
         * that exceed acceptable limits. The ect (estimated completion time) parameter indicates
         * the problematic completion time.
         *
         * @param as  the list of scheduled assignments that caused the timing violation
         * @param ect the estimated completion time that was exceeded
         */
        public record BadAssignements(List<ScheduledAssignment> as, int ect) implements Cut {

            public String toString() {
                StringBuilder st = new StringBuilder(String.format("CUT[BadAss] ECT = %d:", ect));
                for (ScheduledAssignment p : as) {
                    st.append(String.format("\n\tearlier than %d of dataset %d on node %d or less than %d tasks",
                            p.e(), p.a.d(), p.a.n(), p.a.k()));
                }
                return st.toString();
            }
        }


    //    public record StrictlyTooLate(int d, int n, int e, int ft) implements Cut {
    //        @Override
    //        public String toString() {
    //            return String.format("CUT: Transfer earlier than %d of dataset %d on node %d (let = %d)", e, d, n, ft);
    //        }
    //    }

    //    public record TooLate(int d, int n, int e, int ft, int k) implements Cut {
    //        @Override
    //        public String toString() {
    //            return String.format("CUT: Transfer earlier than %d of dataset %d on node %d (let = %d) or less than %d tasks", e, d, n, ft, k);
    //        }
    //    }

    }

    /**
     * Implements Step 1 of the three-step scheduling algorithm: Dataset Splitting.
     * This class uses the ChocoSolver constraint programming library to distribute dataset tasks
     * across available nodes while respecting various constraints.
     * <p>
     * The goal of this step is to determine how many tasks from each dataset should be assigned
     * to each node. This is a fundamental decision that affects the subsequent scheduling steps.
     * The problem is modeled as a constraint satisfaction problem (CSP) where:
     * - Each dataset's tasks must be completely assigned across nodes
     * - The number of tasks assigned to a node cannot exceed the dataset's total tasks
     * - Cuts from previous iterations are respected to avoid infeasible solutions
     */
    public static class DatasetSplittingProblem {

        /**
         * Solves the dataset splitting problem using constraint programming.
         * <p>
         * This method creates a ChocoSolver model to determine the optimal distribution of dataset tasks
         * across nodes. It can operate in two modes:
         * <ul>
         *   <li><b>SAT mode (sat=true):</b> Finds any feasible solution</li>
         *   <li><b>Optimization mode (sat=false):</b> Finds the optimal solution that minimizes
         *       the sum of transfer times across all assignments</li>
         * </ul>
         * <p>
         * The method respects cuts from previous iterations to avoid repeating infeasible configurations.
         *
         * @param nb_data                 the number of datasets
         * @param data_sizes              array containing the size (in MB) of each dataset
         * @param works                   2D array where works[i][j] represents the work amount for the j-th task of dataset i
         * @param starting_times          array containing the arrival/start time of each dataset
         * @param nb_nodes                the number of available nodes
         * @param bandwidths              array containing the bandwidth (in MB/s) of each node
         * @param cpus                    array containing the CPU power of each node
         * @param alpha                   the duration of tasks processing should be greater or equal to alpha * transfer duration
         * @param prevScheduledAssignment the assignments from the previous iteration (used for hints)
         * @param cuts                    the list of cuts to respect from previous iterations
         * @param sat                     if true, find any feasible solution; if false, find the optimal solution
         * @param iteration               current iteration (for randomness)
         * @return a list of assignments representing how dataset tasks are distributed across nodes
         */
        public static List<Commons.Assignment> runScheduler(
                int nb_data, int[] data_sizes, int[][] works, int[] starting_times,
                int nb_nodes, int[] bandwidths, double[] cpus, double alpha,
                List<Commons.Assignment> prevScheduledAssignment,
                List<Commons.Cut> cuts,
                boolean sat,
                int iteration,
                long timeLimitMs,
                int[] storage_capacity) {
            final int CPU_UNIT = 1; // to scale cpu speeds
            // compute an upper bound on makespan (same idea as python)
            long makespanLong = 0;

            long sumData = 0;
            for (int s : data_sizes) sumData += s;

            int minBandwidth = Integer.MAX_VALUE;
            for (int b : bandwidths) if (b < minBandwidth) minBandwidth = b;

            makespanLong = sumData / Math.max(1, minBandwidth);

            int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE);

            boolean printData = false;
            if (printData) {
                System.out.println("Computed makespan upper bound: " + makespan);
                // print inputs (summary)
                System.out.println("DATA");
                for (int i = 0; i < nb_data; i++) {
                    System.out.println(" Data " + i + ": size=" + data_sizes[i] + " MB, works=" + Arrays.toString(works[i]) + " arrival=" + starting_times[i]);
                }
                System.out.println("NODES");
                for (int j = 0; j < nb_nodes; j++) {
                    System.out.println(" Node " + j + ": bandwidth=" + bandwidths[j] + " MB/s");
                }

                System.out.printf("makespan = %d;%n", makespan);
                System.out.printf("nb_data = %d;%n", nb_data);
                System.out.printf("nb_nodes = %d;%n", nb_nodes);

                System.out.printf("data_sizes = %s;%n", Arrays.toString(data_sizes));
                System.out.printf("nb_works = %s;%n", Arrays.toString(Arrays.stream(works).mapToInt(arr -> arr.length).toArray()));
                System.out.printf("work_duration = %s;%n", Arrays.toString(Arrays.stream(works).map(w -> w[0]).toArray()));
                System.out.printf("bandwidths = %s;%n", Arrays.toString(bandwidths));
            }
            // ----- MODEL -----
            Model model = new Model("Bag of Tasks fr.flowtime.Scheduling (Java)",
                    Settings.dev().setLCG(false).setWarnUser(false));

            // ----- CONSTRAINTS -----
            IntVar[][] counters = new IntVar[nb_data][nb_nodes];
            for (int i = 0; i < nb_data; i++) {
                int[] wl = works[i];
                for (int j = 0; j < nb_nodes; j++) {
                    IntIterableRangeSet values;
                    if (data_sizes[i] > storage_capacity[j]) {
                        // Dataset i physically can never fit on node j at all (even alone) --
                        // no task of i can ever be assigned there. Mirrors MainOnline.java's
                        // validNodesList filter.
                        values = new IntIterableRangeSet(0);
                    } else {
                        values = new IntIterableRangeSet((int) Math.ceil((data_sizes[i] * alpha) / (bandwidths[j] * works[i][0] * cpus[j])), IntIterableRangeSet.MAX);
                        values.removeBetween(wl.length + 1, IntIterableRangeSet.MAX);
                        values.add(0);
                    }
                    counters[i][j] = model.intVar("c_" + i + "_" + j, values.toArray());
                }
                model.sum(counters[i], "=", wl.length).post(); // eq (8)
            }

            // manage cuts
            for (Commons.Cut c : cuts) {
                if (c instanceof Commons.BadAssignements(List<Commons.ScheduledAssignment> as, int ect)) {
                    //
                    BoolVar[] bs = new BoolVar[as.size()];
                    int k = 0;
                    for (Commons.ScheduledAssignment a : as) {
                        bs[k++] = model.isLeq(counters[a.a().d()][a.a().n()], a.a().k() - 1);
                    }
                    model.addClausesBoolOrArrayEqualTrue(bs);
                    // todo: same goes for any node with worth bandwidth and worth cpu
                }
            }
            Solver solver = model.getSolver();
    //        model.displayVariableOccurrences();
    //        model.displayPropagatorOccurrences();
            List<Commons.Assignment> E = new ArrayList<>();
    //            solver.showShortStatistics();
            IntVar[] vars = ArrayUtils.flatten(counters);
            solver.setSearch(
                    Search.lastConflict(
                            Search.intVarSearch(new RandomVar<>(iteration, vars), new IntDomainMedian(), vars))
            );
            solver.limitTime(Math.max(50, timeLimitMs) + "ms");
    //        solver.setRestartOnSolutions();
            boolean printSol = false;
            solver.plugMonitor((IMonitorSolution) () -> {
                E.clear();
                for (int i = 0; i < nb_data; i++) {
                    for (int j = 0; j < nb_nodes; j++) {
                        if (counters[i][j].getValue() > 0) {
                            Commons.Assignment a = new Commons.Assignment(i, j,
                                    counters[i][j].getValue());
                            if (printSol) System.out.println(a);
                            E.add(a);
                        }
                    }
                }
            });
            if (sat) {
                solver.findSolution();
            } else {
                int objectiveKind = 1;
                IntVar objective;
                switch (objectiveKind) {
                    case 0:
                        objective = model.count("Min(nb transfers)", 0, ArrayUtils.flatten(counters));
                        break;
                    case 1:
                    default:
                        IntVar[][] condEndTimes = new IntVar[nb_data][nb_nodes];
                        for (int i = 0; i < nb_data; i++) {
                            for (int j = 0; j < nb_nodes; j++) {
                                int d = (int) Math.ceil((double) data_sizes[i] / (double) bandwidths[j]);
                                condEndTimes[i][j] = model.intView(d, model.isEq(counters[i][j], 0).not(), 0);
    //                            condEndTimes[i][j] = model.intView(d, counters[i][j], 0);
                            }
                        }
                        objective = model.sum("Min(sum of transfers)", Arrays.stream(ArrayUtils.flatten(condEndTimes)).toArray(IntVar[]::new));
                        break;
                }
                solver.plugMonitor((IMonitorSolution) () -> {
                    if (printSol) {
                        System.out.printf("Solution %d found (%d)\n", solver.getSolutionCount(), objective.getValue());
                    }
                    // (logging removed)
                });
                solver.findOptimalSolution(objective, false);
                // (logging removed)
            }
            return E;
        }
    }

    /**
     * Implements an alternative approach to Step 2 of the three-step scheduling algorithm:
     * Dataset Assignment for a specific node.
     * This class uses the ChocoSolver constraint programming library to schedule data transfers
     * for a single node, considering only the datasets assigned to that node.
     * <p>
     * Adapted for online/dynamic use in the simulator-for-CSP-model project (see MainOnlineThreeStep):
     * a transfer to this node can never start before {@code nodeStartingTime} (when the node
     * actually becomes free, computed on the Python side from ongoing transfers/work), and a
     * dataset already resident on this node from a previous solve is treated as a near-free
     * (duration 1) reuse instead of a fresh transfer, via {@link #transferTime}.
     */
    public static class DatasetAssignmentProblemNodeJ {

        /**
         * Transfer duration for dataset i to node j: 1 (a minimal, non-zero "reuse" cost) if the
         * data is already resident there from a previous solve, otherwise size/bandwidth (ceil).
         * Mirrors the same helper in MainOnline.java/MainIncremental.java.
         */
        public static double transferTime(int dataIndex, int nodeIndex, int dataSize, int bandwidth, int[][] replicas_location) {
            for (int val : replicas_location[dataIndex]) {
                if (val == nodeIndex) {
                    return 1;
                }
            }
            return (double) dataSize / bandwidth;
        }

        /**
         * Schedules data transfers for a specific node based on the assignments from Step 1.
         *
         * @param node_j            the specific node index to schedule transfers for
         * @param nb_data           the number of datasets
         * @param data_sizes        array containing the size (in MB) of each dataset
         * @param nodeStartingTime  earliest time (relative to "now") this node can start a new transfer
         * @param works             2D array where works[i][j] represents the work amount for the j-th task of dataset i
         * @param nb_nodes          the total number of available nodes
         * @param bandwidths        array containing the bandwidth (in MB/s) of each node
         * @param replicas_location for each dataset i, the list of node ids where it is already resident
         * @param assignment        the assignments from Step 1 (determines how many tasks from each dataset go to each node)
         * @param cuts              the list of cuts to respect from previous iterations
         * @param step2sat          if true, find any feasible solution; if false, find the optimal solution
         * @return a list of scheduled assignments with timing information for assignments on node j
         */
        public static List<Commons.ScheduledAssignment> runScheduler(
                int node_j,
                int nb_data, int[] data_sizes, int nodeStartingTime, int[][] works,
                int nb_nodes, int[] bandwidths,
                int[][] replicas_location,
                List<Commons.Assignment> assignment,
                List<Commons.Cut> cuts,
                boolean step2sat,
                long timeLimitMs) {
            // compute an upper bound on makespan (same idea as python)
            long makespanLong = 0;

            int minBandwidth = bandwidths[node_j];
            for (Commons.Assignment as : assignment) {
                makespanLong += (int) Math.ceil(transferTime(as.d(), node_j, data_sizes[as.d()], Math.max(1, minBandwidth), replicas_location));
            }

            int makespan = (int) Math.min(makespanLong + nodeStartingTime, Integer.MAX_VALUE);

            // ----- MODEL -----
            Model model = new Model("Bag of Tasks Scheduling (Java)",
                    Settings.dev().setLCG(false).setWarnUser(false));

            // load assignment computed by problem A
            int[][] counters = new int[nb_data][nb_nodes];
            for (Commons.Assignment a : assignment) {
                counters[a.d()][a.n()] = a.k();
            }

            // Arrays for transfer tasks and heights
            Task[][] transferTasks = new Task[nb_nodes][nb_data];

            // Create transfer tasks: one per (node, data)
            for (int i = 0; i < nb_data; i++) {
                if (counters[i][node_j] == 0) {
                    transferTasks[node_j][i] = null;
                } else {
                    boolean isResident = Arrays.stream(replicas_location[i]).anyMatch(n -> n == node_j);
                    int d = (int) Math.ceil(transferTime(i, node_j, data_sizes[i], bandwidths[node_j], replicas_location));
                    IntVar s;
                    IntVar end;
                    if (isResident) {
                        // Already physically on this node: pin the (near-instant) "transfer" start
                        // to exactly nodeStartingTime instead of leaving it a free variable up to
                        // makespan -- otherwise the solver could push it arbitrarily late to shrink
                        // the storage-occupancy window Step 3 later charges against capacity (same
                        // free-variable bug already fixed in MainOnline.java/MainIncremental.java).
                        s = model.intVar("start_transfer_d" + i + "_n" + node_j, nodeStartingTime, nodeStartingTime, true);
                        end = model.intVar("end_transfer_d" + i + "_n" + node_j, nodeStartingTime + d, nodeStartingTime + d, true);
                    } else {
                        s = model.intVar("start_transfer_d" + i + "_n" + node_j, nodeStartingTime, makespan, true);
                        end = model.intVar("end_transfer_d" + i + "_n" + node_j, nodeStartingTime + d, makespan, true);
                    }
                    IntVar durationVar = model.intVar(d);
                    Task t = new Task(s, durationVar, end);
                    transferTasks[node_j][i] = t;
                }
            }

            // ----- CONSTRAINTS -----
            // Cumulative constraints for transfers on each node (capacity = 1)
            List<Task> tasksForNode = new ArrayList<>();
            for (int i = 0; i < nb_data; i++) {
                if (transferTasks[node_j][i] != null) {
                    tasksForNode.add(transferTasks[node_j][i]);
                }
            }
            if (!tasksForNode.isEmpty()) {
                model.cumulative(
                        tasksForNode.toArray(new Task[0]),
                        IntStream.range(0, tasksForNode.size()).mapToObj(i -> model.intVar(1)).toArray(IntVar[]::new),
                        model.intVar(1)).post();
            }

            // manage cuts
            cuts:
            for (Commons.Cut c : cuts) {
                if (c instanceof Commons.BadAssignements(
                        List<Commons.ScheduledAssignment> as, int ect
                )) {
                    BoolVar[] bs = new BoolVar[as.size()];
                    int k = 0;
                    for (Commons.ScheduledAssignment a : as) {
                        if (counters[a.a().d()][a.a().n()] <= a.a().k() - 1) break cuts;
                        bs[k++] = model.isLeq(transferTasks[a.a().n()][a.a().d()].getEnd(), a.e() - 1);
                    }
                    model.addClausesBoolOrArrayEqualTrue(bs);
                }
            }

            BoolVar[] prec = new BoolVar[(tasksForNode.size() * (tasksForNode.size() - 1)) / 2];
            for (int i = 0, k = 0; i < tasksForNode.size() - 1; i++) {
                for (int j = i + 1; j < tasksForNode.size(); j++, k++) {
                    prec[k] = model.boolVar("p_" + i + "<" + j);
                    model.impXrelYC(tasksForNode.get(i).getEnd(), "<=", tasksForNode.get(j).getStart(), 0, prec[k]);
                    model.impXrelYC(tasksForNode.get(j).getEnd(), "<=", tasksForNode.get(i).getStart(), 0, prec[k].not());
                }
            }
            if (prec.length == 0) {
                prec = new BoolVar[]{model.boolVar(true)};
            }

            Solver solver = model.getSolver();
            solver.setSearch(
                    Search.minDomUBSearch(prec),
                    Search.minDomLBSearch(tasksForNode.stream().map(Task::getEnd).toArray(IntVar[]::new)));
            solver.limitTime(Math.max(50, timeLimitMs) + "ms");

            List<Commons.ScheduledAssignment> E = new ArrayList<>();
            solver.plugMonitor((IMonitorSolution) () -> {
                E.clear();
                for (Commons.Assignment a : assignment) {
                    Commons.ScheduledAssignment sa = new Commons.ScheduledAssignment(a,
                            transferTasks[a.n()][a.d()].getEnd().getValue(),
                            transferTasks[a.n()][a.d()].getDuration().getValue());
                    E.add(sa);
                }
            });
            if (step2sat) {
                solver.findSolution();
            } else {
                List<IntVar> condEndTimes = new ArrayList<>();
                for (int i = 0; i < nb_data; i++) {
                    for (int j = 0; j < nb_nodes; j++) {
                        if (counters[i][j] > 0) {
                            condEndTimes.add(transferTasks[j][i].getEnd());
                        }
                    }
                }
                IntVar objective = model.max("obj", condEndTimes.toArray(new IntVar[0]));
                int lowerBoundMakespan = calculatePreemptiveSchrageLowerBound(node_j, nb_data, data_sizes, nodeStartingTime, bandwidths, replicas_location, assignment, counters);
                objective.ge(lowerBoundMakespan).post();
                solver.findOptimalSolution(objective, false);
            }
            return E;
        }

        /**
         * Calculates the lower bound of makespan using preemptive Schrage relaxation.
         * For the 1|r_i|F_max problem, the preemptive relaxation gives a lower bound
         * on the optimal non-preemptive makespan.
         */
        private static int calculatePreemptiveSchrageLowerBound(
                int node_j,
                int nb_data,
                int[] data_sizes,
                int nodeStartingTime,
                int[] bandwidths,
                int[][] replicas_location,
                List<Commons.Assignment> assignment,
                int[][] counters) {

            List<TaskInfo> tasks = new ArrayList<>();

            for (int i = 0; i < nb_data; i++) {
                if (counters[i][node_j] > 0) {
                    int transferDuration = (int) Math.ceil(transferTime(i, node_j, data_sizes[i], bandwidths[node_j], replicas_location));
                    tasks.add(new TaskInfo(nodeStartingTime, transferDuration));
                }
            }

            if (tasks.isEmpty()) {
                return 0;
            }

            tasks.sort(Comparator.comparingInt(TaskInfo::releaseDate));

            int currentTime = 0;
            int maxCompletionTime = 0;

            for (TaskInfo task : tasks) {
                int startTime = Math.max(currentTime, task.releaseDate());
                int completionTime = startTime + task.processingTime();

                maxCompletionTime = Math.max(maxCompletionTime, completionTime);
                currentTime = completionTime;
            }

            return maxCompletionTime;
        }

        private record TaskInfo(int releaseDate, int processingTime) {
        }

    }

    /**
     * Implements Step 3 of the three-step scheduling algorithm: Task Allocation.
     * <p>
     * Adapted for online/dynamic use in the simulator-for-CSP-model project (see MainOnlineThreeStep):
     * - a task can never start before {@code nodeStartingTime} (when the node's compute actually
     *   becomes free), in addition to the usual "after its data transfer ends" constraint;
     * - {@code flowTimeOffset[i]} is used instead of a raw arrival time: pass
     *   {@code -job.timelasped} (time already elapsed since the job's real arrival) so that
     *   {@code flow = jobEnd - flowTimeOffset = jobEnd + timelasped} matches the true, absolute
     *   flow time, exactly like MainOnline.java's own elapsedTime handling. Unlike a real arrival
     *   time, this can legitimately be negative for a job that has been running a while, so it
     *   must NOT be used as a scheduling lower bound (nodeStartingTime and the transfer's own end
     *   time already cover that).
     * - kept ONE_TASK_PER_DATASET=false: the simulator dispatches work one task at a time (see
     *   works.csv / master_node's per-task queue), so each task needs its own row/timing, not one
     *   merged block per dataset.
     */
    public static class TaskAllocationProblemNodeJ {

        private static final boolean ONE_TASK_PER_DATASET = false;

        /**
         * Schedules task execution on a specific node based on scheduled assignments from Step 2.
         *
         * @param node_j               the specific node index to schedule tasks for
         * @param nb_data              the number of datasets
         * @param works                2D array where works[i][j] represents the work amount for the j-th task of dataset i
         * @param cpus                 array containing the CPU power of each node
         * @param flowTimeOffset       per-dataset offset such that flow = jobEnd - flowTimeOffset (pass -timelasped)
         * @param nodeStartingTime     earliest time (relative to "now") this node's compute can start a new task
         * @param scheduledAssignments the scheduled assignments from Step 2 (determines when data transfers complete)
         * @param bestKnwoMaxFlowTime  the best known maximum flow time from previous iterations (used as upper bound)
         * @param step3sat             if true, find any feasible solution; if false, find the optimal solution
         * @return a Result object (plans, cuts, maxFlow, sumFlow)
         */
        public static Commons.Result runScheduler(
                int node_j,
                int nb_data,
                int[][] works,
                double[] cpus,
                int[] flowTimeOffset,
                int nodeStartingTime,
                List<Commons.ScheduledAssignment> scheduledAssignments,
                int bestKnwoMaxFlowTime,
                boolean step3sat,
                long timeLimitMs,
                int[] data_sizes,
                int storageCapacity) {
            if (scheduledAssignments.isEmpty())
                return new Commons.Result(Collections.emptyList(), Collections.emptyList(), -1, 0);
            final int CPU_UNIT = 1; // to scale cpu speeds
            // compute an upper bound on makespan (same idea as python)
            long makespanLong = 0;

            long totalWork = 0;
            for (int[] wl : works) for (int w : wl) totalWork += w;

            double maxCpu = 0;
            for (double c : cpus) if (c > maxCpu) maxCpu = c;

            makespanLong += (long) (totalWork * CPU_UNIT * Math.max(1, maxCpu));
            makespanLong += nodeStartingTime;
            makespanLong *= 2;

            int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE);

            // ----- MODEL -----
            Model model = new Model("Bag of Tasks Scheduling on Node " + node_j + "(Java)",
                    Settings.dev().setLCG(false).setWarnUser(false));

            IntVar[][] jobStarts = new IntVar[nb_data][];
            int[][] jobDurations = new int[nb_data][];
            IntVar[][] jobEnds = new IntVar[nb_data][];
            IntVar[] jobFlow = new IntVar[nb_data];
            List<Task> tasks = new ArrayList<>();
            // Storage occupancy: dataset i sits on this node's disk from the moment its transfer
            // here starts (fixed by Step 2, not a decision here) until the last task using it on
            // this node finishes (release, a decision of THIS model) -- mirrors MainOnline.java's
            // storage cumulative constraint, just split across steps since Step 2 already committed
            // to the transfer timing before this model runs.
            List<Task> storageTasks = new ArrayList<>();
            List<IntVar> storageHeights = new ArrayList<>();
            int sumDurations = 0;
            int lst = 0;
            for (int i = 0; i < nb_data; i++) {
                final int fi = i;
                Optional<Commons.ScheduledAssignment> assignment = scheduledAssignments.stream().filter(a -> a.a().d() == fi).findAny();
                if (assignment.isEmpty()) continue;

                int transferEnd = Math.max(assignment.get().e(), nodeStartingTime);
                int transferStart = assignment.get().e() - assignment.get().td();
                if (lst < transferEnd) {
                    lst = transferEnd;
                }

                int executionDuration = 0;
                int nb_works = assignment.get().a().k();
                jobStarts[i] = new IntVar[nb_works];
                jobDurations[i] = new int[nb_works];
                jobEnds[i] = new IntVar[nb_works];

                int w = works[i][0]; // every task of a dataset has the same duration in this project
                int duration = (int) (w * cpus[node_j]);
                for (int k = 0; k < nb_works; k++) {
                    jobStarts[i][k] = model.intVar("start_work_d" + i + "_w" + k, transferEnd, makespan, true);
                    sumDurations += duration;
                    jobDurations[i][k] = duration;
                    executionDuration += duration;
                    jobEnds[i][k] = model.intView(1, jobStarts[i][k], jobDurations[i][k]);
                    tasks.add(new Task(jobStarts[i][k], duration, jobEnds[i][k]));
                    if (k > 0) {
                        jobEnds[i][k - 1].eq(jobStarts[i][k]).post();
                    }
                }

                if (flowTimeOffset[i] + executionDuration > bestKnwoMaxFlowTime) {
                    return new Commons.Result(Collections.emptyList(),
                            List.of(new Commons.BadAssignements(scheduledAssignments, flowTimeOffset[i] + executionDuration)), -1, 0);
                }
                jobFlow[i] = model.intView(1, model.max("flow_time_d" + i, jobEnds[i]), -flowTimeOffset[i]);
                model.arithm(jobFlow[i], "<", bestKnwoMaxFlowTime).post();

                // release must be able to reach jobEnds[i]'s true upper bound (jobStarts[i][k] can
                // itself be as late as makespan, so jobEnds[i][k] = start + duration can reach
                // makespan + duration) -- capping release's domain at plain makespan made the
                // model spuriously infeasible whenever a task's actual end fell in that
                // (makespan, makespan+duration] gap, since max(release, jobEnds[i]) could then
                // never be satisfied.
                int releaseUpperBound = makespan + duration;
                IntVar release = model.intVar("storage_release_d" + i + "_n" + node_j, transferStart, releaseUpperBound, true);
                model.max(release, jobEnds[i]).post();
                IntVar storageDuration = model.intVar("storage_duration_d" + i + "_n" + node_j, 0, releaseUpperBound, true);
                storageTasks.add(new Task(model.intVar(transferStart), storageDuration, release));
                storageHeights.add(model.intVar(data_sizes[i]));
            }
            for (Task t : tasks) {
                t.getEnd().le(lst + sumDurations).post();
            }

            model.cumulative(
                    tasks.toArray(new Task[0]),
                    tasks.stream().map(t -> model.intVar(1)).toArray(IntVar[]::new),
                    model.intVar(1)
            ).post();

            if (!storageTasks.isEmpty()) {
                model.cumulative(
                        storageTasks.toArray(new Task[0]),
                        storageHeights.toArray(new IntVar[0]),
                        model.intVar(storageCapacity)
                ).post();
            }

            BoolVar[] prec = new BoolVar[(tasks.size() * (tasks.size() - 1)) / 2];
            for (int i = 0, k = 0; i < tasks.size() - 1; i++) {
                for (int j = i + 1; j < tasks.size(); j++, k++) {
                    prec[k] = model.boolVar("p_" + i + "<" + j);
                    model.impXrelYC(tasks.get(i).getEnd(), "<=", tasks.get(j).getStart(), 0, prec[k]);
                    model.impXrelYC(tasks.get(j).getEnd(), "<=", tasks.get(i).getStart(), 0, prec[k].not());
                }
            }
            if (prec.length == 0) {
                prec = new BoolVar[]{model.boolVar(true)};
            }

            IntVar objective = model.max("objective",
                    Arrays.stream(jobFlow).filter(Objects::nonNull).toArray(IntVar[]::new));
            model.setObjective(false, objective);
            Solver solver = model.getSolver();
            solver.setSearch(
                    Search.minDomUBSearch(prec),
                    Search.minDomLBSearch(Arrays.stream(jobFlow).filter(Objects::nonNull).toArray(IntVar[]::new)));
            solver.limitTime(Math.max(50, timeLimitMs) + "ms");
            List<Commons.Plan> P = new ArrayList<>();
            List<Commons.Cut> C = new ArrayList<>();
            int[] maxFlowTime = {-1, 0};
            solver.plugMonitor((IMonitorSolution) () -> {
                P.clear();
                C.clear();
                for (int i = 0; i < nb_data; i++) {
                    if (jobStarts[i] != null) {
                        for (int k = 0; k < jobStarts[i].length; k++) {
                            Commons.Plan p = new Commons.Plan(i, node_j, k, jobStarts[i][k].getValue(), jobEnds[i][k].getValue());
                            P.add(p);
                        }
                        int flowtime = jobFlow[i].getValue();
                        maxFlowTime[0] = Math.max(flowtime, maxFlowTime[0]);
                        maxFlowTime[1] += flowtime;
                    }
                }
                C.add(new Commons.BadAssignements(scheduledAssignments, maxFlowTime[0]));
            });
            if (step3sat) {
                solver.limitSolution(1);
            }
            while (solver.solve()) ;
            if (solver.getSolutionCount() == 0) {
                return new Commons.Result(Collections.emptyList(),
                        List.of(new Commons.BadAssignements(scheduledAssignments, Integer.MAX_VALUE)), -1, 0);
            }
            return new Commons.Result(P, C, maxFlowTime[0], maxFlowTime[1]);
        }
    }

}
