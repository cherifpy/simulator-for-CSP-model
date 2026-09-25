//package simulator.utils.CSPModel;
package main;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

import static org.chocosolver.solver.search.strategy.Search.*;

import gnu.trove.TIntCollection;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.search.limits.FailCounter;
import org.chocosolver.solver.search.loop.lns.neighbors.*;
import org.chocosolver.solver.search.restart.GeometricalCutoff;
import org.chocosolver.solver.search.restart.InnerOuterCutoff;
import org.chocosolver.solver.search.restart.LubyCutoff;
import org.chocosolver.solver.search.restart.Restarter;
import org.chocosolver.solver.search.strategy.BlackBoxConfigurator;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainBest;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainLast;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainMax;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainMin;
import org.chocosolver.solver.search.strategy.selectors.variables.InputOrder;
import org.chocosolver.solver.search.strategy.strategy.*;
import org.chocosolver.solver.variables.*;


import java.util.*;
import java.io.*;

import org.json.JSONArray;
import org.json.JSONObject;

import org.chocosolver.solver.Settings;

import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.search.loop.lns.neighbors.INeighbor;
import org.chocosolver.solver.search.loop.lns.neighbors.SequenceNeighborhood;
import org.chocosolver.solver.search.strategy.selectors.values.*;
import org.chocosolver.solver.search.strategy.selectors.variables.*;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;
import org.chocosolver.util.sort.ArraySort;
import org.chocosolver.util.tools.ArrayUtils;

import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.constraints.Constraint;
import org.chocosolver.solver.constraints.Propagator;
import org.chocosolver.solver.objective.ParetoMaximizer;
import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.search.limits.ICounter;
import org.chocosolver.solver.search.loop.lns.neighbors.INeighbor;
import org.chocosolver.solver.search.loop.move.Move;
import org.chocosolver.solver.search.restart.GeometricalCutoff;
import org.chocosolver.solver.search.restart.ICutoff;
import org.chocosolver.solver.search.restart.InnerOuterCutoff;
import org.chocosolver.solver.search.restart.LubyCutoff;
import org.chocosolver.solver.search.strategy.decision.RootDecision;
import org.chocosolver.solver.search.strategy.strategy.AbstractStrategy;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Variable;
import org.chocosolver.solver.variables.events.IntEventType;
import org.chocosolver.util.ESat;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MainOnlineMultiObj {

    // Portable base path: the Python caller (utils/modelCSP.py) always launches this JVM with
    // its working directory set to the `simulator/` folder, so paths built from here stay valid
    // wherever that folder is copied (a different machine, a different username/home -- e.g.
    // Grid5000), instead of being hardcoded to one developer's machine.
    private static final String SIMULATOR_DIR = System.getProperty("user.dir");
    private static final String MODEL_INPUTS_DIR = SIMULATOR_DIR + "/utils/model/inputs";
    private static final String MODEL_OUTPUTS_DIR = SIMULATOR_DIR + "/utils/model/outputs";

    public static class MyMoveLNS implements Move {

        /**
         * the strategy required to complete the generated fragment
         */
        protected Move move;
        /**
         * IntNeighbor to used
         */
        protected INeighbor neighbor;
        /**
         * Number of solutions found so far
         */
        protected long solutions;
        /**
         * Indicates if a solution has been loaded
         */
        protected boolean solutionLoaded;
        /**
         * Indicate a restart has been triggered
         */
        private boolean freshRestart;
        /**
         * Restart counter
         */
        protected ICounter counter;
        private final ICutoff restartStrategy;
        /**
         * For restart strategy
         */
        //private final long frequency;

        protected PropLNS prop;

        private boolean canApplyNeighborhood;

        /**
         * Create a move which defines a Large Neighborhood Search.
         *
         * @param move           how the subtree is explored
         * @param neighbor       how the fragment are computed
         * @param restartCounter when a restart should occur
         */
        public MyMoveLNS(Move move, INeighbor neighbor, ICounter restartCounter) {
            this.move = move;
            this.neighbor = neighbor;
            this.counter = restartCounter;
            //this.frequency = counter.getLimitValue();
            this.restartStrategy =
                    //new GeometricalCutoff(counter.getLimitValue(), 1.01);
                    new InnerOuterCutoff(counter.getLimitValue(), 1.01, 1.01);
            //new LubyCutoff(counter.getLimitValue());
            this.solutions = 0;
            this.freshRestart = false;
            this.solutionLoaded = false;
        }

        @Override
        public boolean init() {
            neighbor.init();
            return move.init();
        }

        /**
         * Return false when:
         * <ul>
         * <li>
         * the underlying search has no more decision to provide,
         * </li>
         * </ul>
         * <p>
         * Return true when:
         * <ul>
         * <li>
         * a new neighbor is provided,
         * </li>
         * <li>
         * or a new decision is provided by the underlying decision
         * </li>
         * <li>
         * or the fast restart criterion is met.
         * </li>
         * </ul>
         * <p>
         * Restart when:
         * <ul>
         * <li>
         * a restart criterion is met
         * </li>
         * </ul>
         *
         * @param solver SearchLoop
         * @return true if the decision path is extended
         */
        @Override
        public boolean extend(Solver solver) {
            boolean extend;
            // when a new fragment is needed (condition: at least one solution has been found)
            if (solutions > 0 || solutionLoaded) {
                if (freshRestart) {
                    assert solver.getDecisionPath().size() == 1;
                    assert solver.getDecisionPath().getDecision(0) == RootDecision.ROOT;
                    solver.pushTrail();
                    if (prop == null) {
                        prop = new PropLNS(solver.getModel().intVar(2));
                        new Constraint("LNS", prop).post();
                    }
                    solver.getEngine().propagateOnBacktrack(prop);
                    canApplyNeighborhood = true;
                    freshRestart = false;
                    extend = true;
                } else {
                    // if fast restart is on
                    if (counter.isMet()) {
                        // then is restart is triggered
                        doRestart(solver);
                        extend = true;
                    } else {
                        extend = move.extend(solver);
                    }
                }
            } else {
                extend = move.extend(solver);
            }
            return extend;
        }

        /**
         * Return false when :
         * <ul>
         * <li>
         * move.repair(searchLoop) returns false and neighbor is complete.
         * </li>
         * <li>
         * posting the cut at root node fails
         * </li>
         * </ul>
         * Return true when:
         * <ul>
         * <li>
         * move.repair(searchLoop) returns true,
         * </li>
         * <li>
         * or move.repair(searchLoop) returns false and neighbor is not complete,
         * </li>
         * </ul>
         * <p>
         * Restart when:
         * <ul>
         * <li>
         * a new solution has been found
         * </li>
         * <li>
         * move.repair(searchLoop) returns false and neighbor is not complete,
         * </li>
         * <li>
         * or the fast restart criterion is met
         * </li>
         * </ul>
         *
         * @param solver SearchLoop
         * @return true if the decision path is repaired
         */
        @Override
        public boolean repair(Solver solver) {
            boolean repair = true;
            if (solutions > 0
                    // the second condition is only here for intiale calls, when solutions is not already up to date
                    || solver.getSolutionCount() > 0
                    // the third condition is true when a solution was given as input
                    || solutionLoaded) {
                // the detection of a new solution can only be met here
                if (solutions < solver.getSolutionCount()) {
                    assert solutions == solver.getSolutionCount() - 1;
                    solutions++;
                    solutionLoaded = false;
                    neighbor.recordSolution();
                    doRestart(solver);
                    this.restartStrategy.reset();
                }
                // when posting the cut directly at root node fails
                else if (freshRestart) {
                    repair = false;
                }
                // the current sub-tree has been entirely explored
                else if (!(repair = move.repair(solver))) {
                    // but the neighbor cannot ensure completeness
                    if (!neighbor.isSearchComplete()) {
                        // then a restart is triggered
                        doRestart(solver);
                        repair = true;
                    }
                }
                // or a fast restart is on
                else if (counter.isMet()) {
                    // then is restart is triggered
                    doRestart(solver);
                }
            } else {
                repair = move.repair(solver);
            }
            return repair;
        }

        /**
         * Give an initial solution to begin with if called before executing the solving process
         * or erase the last recorded one otherwise.
         *
         * @param solution a solution to record
         * @param solver   that manages the LNS
         */
        public void loadFromSolution(Solution solution, Solver solver) {
            neighbor.loadFromSolution(solution);
            solutionLoaded = true;
            if (solutions == 0) {
                freshRestart = true;
            } else {
                doRestart(solver);
            }
        }

        @Override
        public void setTopDecisionPosition(int position) {
            move.setTopDecisionPosition(position);
        }

        @Override
        public <V extends Variable> AbstractStrategy<V> getStrategy() {
            return move.getStrategy();
        }

        @Override
        public <V extends Variable> void setStrategy(AbstractStrategy<V> aStrategy) {
            move.setStrategy(aStrategy);
        }

        @Override
        public void removeStrategy() {
            move.removeStrategy();
        }

        /**
         * Extend the neighbor when conditions are met and do the restart
         *
         * @param solver SearchLoop
         */
        private void doRestart(Solver solver) {
            if (!freshRestart) {
                neighbor.restrictLess();
            }
            freshRestart = true;
            long nc = restartStrategy.getNextCutoff();
            counter.overrideLimit(counter.currentValue() + nc);
            //System.out.printf("nc : %d%n", nc);
            solver.restart();
        }

        @Override
        public List<Move> getChildMoves() {
            return Collections.singletonList(move);
        }

        @Override
        public void setChildMoves(List<Move> someMoves) {
            if (someMoves.size() == 1) {
                this.move = someMoves.get(0);
            } else {
                throw new UnsupportedOperationException("Only one child move can be attached to it.");
            }
        }

        class PropLNS extends Propagator<IntVar> {

            PropLNS(IntVar var) {
                super(var);
                this.vars = new IntVar[0];
            }

            @Override
            public int getPropagationConditions(int vIdx) {
                return IntEventType.VOID.getMask();
            }

            @Override
            public void propagate(int evtmask) throws ContradictionException {
                if (canApplyNeighborhood) {
                    canApplyNeighborhood = false;
                    neighbor.fixSomeVariables();
                }
            }

            @Override
            public ESat isEntailed() {
                return ESat.TRUE;
            }
        }
    }

    public class SchedulingWithDiffN {

        public static class TransferConfig {
            int jobIndex;
            int startTime;
            int endTime;
            int nodeIndex;

            public TransferConfig(int jobIndex, int startTime, int endTime, int nodeIndex) {
                this.jobIndex = jobIndex;
                this.startTime = startTime;
                this.endTime = endTime;
                this.nodeIndex = nodeIndex;
            }

        }

        public static class WorkConfig {
            int taskindex;
            int jobIndex;
            int startTime;
            int endTime;
            int nodeIndex;

            public WorkConfig(int taskindex, int jobindex, int startTime, int endTime, int nodeIndex) {
                this.taskindex = taskindex;
                this.jobIndex = jobindex;
                this.startTime = startTime;
                this.endTime = endTime;
                this.nodeIndex = nodeIndex;
            }
        }

        public static class DeletionConfig {
            int jobIndex;
            int nodeIndex;
            int deletionTime;

            public DeletionConfig(int jobIndex, int nodeIndex, int deletionTime) {
                this.jobIndex = jobIndex;
                this.nodeIndex = nodeIndex;
                this.deletionTime = deletionTime;
            }
        }

        public static class SchedulingResult {
            public List<TransferConfig> transfers = new ArrayList<>();
            public List<WorkConfig> worksExec = new ArrayList<>();
            public List<DeletionConfig> deletions = new ArrayList<>();

            public SchedulingResult(List<TransferConfig> transfers, List<WorkConfig> worksExec, List<DeletionConfig> deletions) {
                this.transfers = transfers;
                this.worksExec = worksExec;
                this.deletions = deletions;
            }
        }

        public static void writeNodeConfigCSV(List<MainOnlineMultiObj.NodeConfig> nodes, String path) throws Exception {
            FileWriter writer = new FileWriter(path);

            // header
            writer.write("bandwidth,computation_nodes,energy_consumption,storage_capacity\n");

            // rows
            for (MainOnlineMultiObj.NodeConfig n : nodes) {
                writer.write(
                        n.bandwidth + "," +
                                n.computationNodes + "," +
                                n.energyConsumption + "," +
                                n.storageCapacity + "\n"
                );
            }

            writer.close();
        }

        public static void writeTransferConfigCSV(List<SchedulingWithDiffN.TransferConfig> transfers, String path) throws Exception {
            FileWriter writer = new FileWriter(path);

            // header
            writer.write("job_index,start_time,end_time,node_index\n");
            //System.out.println("Transfers confidguration");
            // rows
            for (TransferConfig t : transfers) {
                writer.write(
                        t.jobIndex + "," +
                                t.startTime + "," +
                                t.endTime + "," +
                                t.nodeIndex + "\n"
                );
                //System.out.println("Transfer for job " + t.jobIndex + " on node " + t.nodeIndex + " from " + t.startTime + " to " + t.endTime);
            }
            
            writer.close();
        }

        public static void writeDeletionConfigCSV(List<SchedulingWithDiffN.DeletionConfig> deletions, String path) throws Exception {
            FileWriter writer = new FileWriter(path);

            // header
            writer.write("job_index,node_index,deletion_time\n");
            // rows
            for (DeletionConfig del : deletions) {
                writer.write(
                        del.jobIndex + "," +
                                del.nodeIndex + "," +
                                del.deletionTime + "\n"
                );
            }

            writer.close();
        }

        public static SchedulingResult runScheduler(List<MainOnlineMultiObj.Job>  jobs,int nb_nodes, int nb_data, int[] data_sizes, int[][] works, int[] bandwidths, double[] cpus, int[] storage_capacity, double[] starting_times, int[][]  replicas_location, Model[] models, int pos, boolean solve, double[] job_arriving_times, double[] node_energy_consumption) {
            final int CPU_UNIT = 1; // to scale cpu speeds

            // starting_times[j] (how long node j stays busy) comes from the Python simulator as
            // a float, computed from real (float) simulation time -- Choco's IntVars only take
            // ints. A plain (int) cast truncates toward zero, so a node that's actually free at
            // t=45.7 would be treated as free at t=45, letting the model schedule something
            // 0.7 units before the node is really available. Round up AND add a full extra unit
            // of margin on top (per the same float/int mismatch on the Python side, where not
            // every code path already adds its own margin) so this bound is never optimistic.
            int[] nodeStartingTimes = new int[nb_nodes];
            for (int j = 0; j < nb_nodes; j++) {
                nodeStartingTimes[j] = (int) Math.ceil(starting_times[j]) + 1;
            }
            // compute an upper bound on makespan (same idea as python)
            long makespanLong = 0;

            long sumData = 0;
            for (int s : data_sizes) sumData += s;

            int minBandwidth = Integer.MAX_VALUE;
            for (int b : bandwidths) if (b < minBandwidth) minBandwidth = b;

            makespanLong = sumData / Math.max(1, minBandwidth);

            long totalWork = 0;
            for (int[] wl : works) for (int w : wl) totalWork += w;

            double maxCpu = 0;
            for (double c : cpus) if (c > maxCpu) maxCpu = c;

            double maxStartingTime = 0;
            for (double s : starting_times) if (s > maxStartingTime) maxStartingTime = s;

            // maxStartingTime was computed above but never folded in here (silently added as 0) --
            // with enough existing/queued work, a node's own starting_times[j] can already exceed
            // whatever this bound would otherwise be, so every "start_transfer_d..._n..." IntVar
            // built as [starting_times[j], makespan] below ends up with lower > upper -- a Choco
            // SolverException, not a graceful infeasible-result return. Folding it in first
            // guarantees makespan is always at least as large as the latest node/job starting
            // point before the data/compute-volume margin is added on top.
            makespanLong += (long) Math.ceil(maxStartingTime) + 1;
            makespanLong += totalWork * CPU_UNIT * Math.max(1, maxCpu);
            makespanLong *= 2;

            // Was hardcoded to a fixed 10_000s horizon -- fine for light workloads, but silently
            // wrong (not just suboptimal: an outright SolverException, since node/job starting
            // times can then exceed this fixed ceiling) once enough existing jobs/data volume
            // push the real horizon past it. Use the dynamically-computed bound instead.
            int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE);

            // Optional hard node filter: when present, a node NOT listed is completely excluded
            // from this solve's candidates -- no notion of "will be free in X time units", just
            // in/out. Absent file (the default, used by Online/regular Incremental) means every
            // node stays a candidate (subject to the storage-size filter below), unchanged from
            // prior behavior.
            //boolean[] nodeFree = new boolean[nb_nodes];
            //java.util.Arrays.fill(nodeFree, true);
            //try {
            //    String freeNodesText = readFile(MODEL_INPUTS_DIR + "/free_nodes.txt").trim();
            //    if (!freeNodesText.isEmpty()) {
            //        java.util.Arrays.fill(nodeFree, false);
            //        for (String tok : freeNodesText.split(",")) {
            //            if (!tok.trim().isEmpty()) nodeFree[Integer.parseInt(tok.trim())] = true;
            //        }
            //    }
            //} catch (Exception e) {
            //    // File missing/unreadable: keep every node free (no restriction), matching the
            //    // behavior before this filter existed.
            //}

            // Ghost storage entries: data that belongs to jobs NOT part of this solve's batch at
            // all (e.g. a job that already had every task dispatched and dropped out of
            // reconsideration, or -- for Incremental -- literally any other job) but whose bytes
            // are still physically sitting on a node. These aren't schedulable data (no work
            // list, no transfer decision) -- just a known amount of space occupied on a node
            // until a known (or, if not yet decided, indefinite) release time, folded into that
            // node's SAME storage cumulative constraint below so real capacity is never
            // overbooked by something this solve otherwise can't see at all.
            //class GhostStorage {
            //    int nodeId; int size; int deletionTime;
            //    GhostStorage(int n, int s, int d) { nodeId = n; size = s; deletionTime = d; }
            //}
            //List<GhostStorage> ghostStorage = new ArrayList<>();
            //try {
            //    String ghostText = readFile(MODEL_INPUTS_DIR + "/ghost_storage.txt").trim();
            //    if (!ghostText.isEmpty()) {
            //        for (String line : ghostText.split("\n")) {
            //            if (line.trim().isEmpty()) continue;
            //            String[] parts = line.trim().split(",");
            //            int gNode = Integer.parseInt(parts[0].trim());
            //            int gSize = Integer.parseInt(parts[1].trim());
            //            int gDeletion = Integer.parseInt(parts[2].trim());
            //            // -1 means "no deletion decided yet for this (job,node)": treat as kept
            //            // indefinitely within this horizon, same safe default as regular data.
            //            ghostStorage.add(new GhostStorage(gNode, gSize, gDeletion < 0 ? makespan : gDeletion));
            //        }
            //    }
            //} catch (Exception e) {
            //    // File missing/unreadable: no ghost entries (matches behavior before this existed).
            //}

            // Absolute simulator clock at the moment this solve was launched, purely for debug
            // prints below -- lets us show times directly comparable to the "start:"/"end:"
            // values Python prints for the final solution (which are also now+local).
            double currentSimTime = 0.0;
            try {
                currentSimTime = Double.parseDouble(readFile(MODEL_INPUTS_DIR + "/current_sim_time.txt").trim());
            } catch (Exception e) {
                // File missing/unreadable: debug prints just show 0 for "now" instead of crashing.
            }

            // Jobs the Python side decided are too costly to move (e.g. a large dataset already
            // resident somewhere): comma-separated job indices (same nb_data indexing as
            // data_sizes/replicas_location), written to frozen_jobs.txt by
            // _schedulingUsingJavaCSP_impl. A frozen job's tasks stay confined to nodes it's
            // ALREADY resident on (no new node, so no new transfer -- see validNodes below), and
            // its transferHeights are fixed to its current residency instead of left free (see
            // the transfer-task loop below) -- both trims real search space AND guarantees no
            // network cost from moving it. File missing/empty (the default): no job is frozen,
            // identical to behavior before this existed.
            boolean[] isFrozen = new boolean[nb_data];
            try {
                String frozenText = readFile(MODEL_INPUTS_DIR + "/frozen_jobs.txt").trim();
                if (!frozenText.isEmpty()) {
                    for (String tok : frozenText.split(",")) {
                        int idx = Integer.parseInt(tok.trim());
                        if (idx >= 0 && idx < nb_data) isFrozen[idx] = true;
                    }
                }
            } catch (Exception e) {
                // File missing/unreadable: no job frozen (matches behavior before this existed).
            }

            // ----- MODEL -----
            Model model = new Model("Bag of Tasks Scheduling (Java)");
            /*Settings.dev()
                    .setLCG(false)
                    .setWarnUser(true));
            */    
            // Arrays for transfer tasks and heights
            Task[][] transferTasks = new Task[nb_nodes][nb_data];
            BoolVar[][] transferHeights = new BoolVar[nb_nodes][nb_data];

            // Per-(node,data) candidate transfer's energy cost, computed as an integer constant
            // up front (mirrors Tracker.log_transfer / compute_transfer_energy exactly: sender +
            // receiver*duration + network) since data_sizes[i]/bandwidths[j]/node_energy_consumption[j]
            // are all already-known constants once (i,j) is fixed -- the only thing decided by
            // the solver is WHETHER this candidate is selected (transferHeights[j][i]), not its
            // cost. Filled in alongside `d` below; used to build the energy objective further down.
            int[][] energyContrib = new int[nb_nodes][nb_data];
            double senderEnergy = 0.0, networkEnergy = 0.0;
            try {
                List<String> energyLines = readLines(MODEL_INPUTS_DIR + "/energy_config.txt");
                if (energyLines.size() >= 2) {
                    senderEnergy = Double.parseDouble(energyLines.get(0).trim());
                    networkEnergy = Double.parseDouble(energyLines.get(1).trim());
                }
            } catch (Exception e) {
                // File missing/unreadable: energy terms default to 0 (matches config.json's own
                // 0.0 defaults when master_energy_consumption/network_energy_per_transfer are unset).
            }

            // Create transfer tasks: one per (node, data)

            boolean free_only = false;
            boolean use_the_node = true;
            for (int j = 0; j < nb_nodes; j++) {
                

                for (int i = 0; i < nb_data; i++) {

                    IntVar s; // = model.intVar("start_transfer_d" + i + "_n" + j, (int) starting_times[j], makespan,true);
                    
                    int d; // = (int) Math.ceil(transferTime(i, j, data_sizes[i], bandwidths[j], replicas_location));
                    //System.out.println("Transfer time for data " + i + " on node " + j + ": " + d + "starting_time: " + starting_times[j]);
                    //IntVar durationVar = model.intVar(d);
                    IntVar end;// = model.intVar("end_transfer_d" + i + "_n" + j, (int) starting_times[j] + d, makespan,true);
                    
                    // s/end must stay internally consistent (s + d = end) regardless of which
                    // branch set h, or the Task below is contradictory and the WHOLE model
                    // becomes infeasible the moment any single node is too small for any single
                    // job -- even though h=false already means this pair can never be selected.
                    final int jForResidentCheck = j;
                    boolean isResident = Arrays.stream(replicas_location[i]).anyMatch(n -> n == jForResidentCheck);

                    BoolVar h;
                    if (data_sizes[i] > storage_capacity[j]) {
                        h = model.boolVar("height_transfer_d" + i + "_n" + j, false);
                    } else if (isFrozen[i] && !isResident) {
                        // Frozen and NOT already here: no new replica reaches this node. Cannot
                        // also force h=true on every node it's ALREADY resident on below -- h is
                        // tied by reification to counters[j]>=1 (a task actually landing there),
                        // and sum(counters) is constrained to equal this job's own task count
                        // (wl.length) a few lines down. A job can easily be resident on MORE
                        // nodes than it has tasks (replicas accumulate across many replans), so
                        // forcing every one of those true would force sum(counters) past
                        // wl.length -- outright infeasible. Leaving already-resident nodes free
                        // (the else branch) lets the normal mechanism decide which of them still
                        // keep a task this round, exactly like any non-frozen job's unused
                        // replicas -- freezing only ever forbids reaching a NEW node.
                        h = model.boolVar("height_transfer_d" + i + "_n" + j, false);
                    } else {
                        h = model.boolVar("height_transfer_d" + i + "_n" + j);
                    }
                    if (isResident) {
                        // Data is already physically on this node from a previous solve: pin its
                        // storage-occupancy start to "now" (nodeStartingTimes[j]) instead of leaving
                        // it a free variable the solver could push arbitrarily far into the future to
                        // manufacture spare capacity for other placements -- that free-start gap is
                        // what let already-resident data go uncounted against real storage usage.
                        s = model.intVar("start_transfer_d" + i + "_n" + j, nodeStartingTimes[j], nodeStartingTimes[j], true);
                        d = 1;
                        end = model.intVar("end_transfer_d" + i + "_n" + j, nodeStartingTimes[j] + d, nodeStartingTimes[j] + d, true);
                    } else {
                        // A data item can't start transferring before its own job has actually
                        // arrived -- nodeStartingTimes[j] alone only bounds this by when the NODE
                        // is free, which says nothing about the JOB itself. job_arriving_times[i]
                        // is 0 for the ordinary case (job already arrived relative to this solve's
                        // own "now"), so this is a no-op there; it only bites for a joint solve
                        // over jobs with staggered real arrival times (e.g. state A's one-shot
                        // build), where it stops the solver from silently scheduling a job's
                        // transfer before local time 0 = its real arrival.
                        int arrivalLb = job_arriving_times == null ? 0 : (int) Math.ceil(job_arriving_times[i]);
                        int lb = Math.max(nodeStartingTimes[j], arrivalLb);
                        s = model.intVar("start_transfer_d" + i + "_n" + j, lb, makespan, true);
                        d = (int) Math.ceil(transferTime(i, j, data_sizes[i], bandwidths[j], replicas_location));
                        end = model.intVar("end_transfer_d" + i + "_n" + j, lb + d, makespan, true);
                    }
                    IntVar durationVar = model.intVar(d);

                    Task t = new Task(s, durationVar, end);
                    transferTasks[j][i] = t;
                    transferHeights[j][i] = h;
                    energyContrib[j][i] = (int) Math.round(senderEnergy + node_energy_consumption[j] * d + networkEnergy);
                }
            }

            // Create work tasks: for each node, each data, each wor
            //
            IntVar[][] jobStarts = new IntVar[nb_data][];
            IntVar[][] jobDurations = new IntVar[nb_data][];
            IntVar[][] jobEnds = new IntVar[nb_data][];
            IntVar[][] jobNodes = new IntVar[nb_data][];

            for (int i = 0; i < nb_data; i++) {
                int[] wl = works[i]; 
                jobStarts[i] = new IntVar[wl.length];
                jobDurations[i] = new IntVar[wl.length];
                jobEnds[i] = new IntVar[wl.length];
                jobNodes[i] = new IntVar[wl.length];

                // Reduction de l'espace de recherche : un noeud dont la capacite de
                // stockage est trop petite pour la donnee i ne peut de toute facon
                // jamais recevoir son transfert (transferHeights fixe a 0 plus haut),
                // donc aucune tache liee a cette donnee ne peut s'y executer non plus.
                // On retire directement ces noeuds du domaine de jobNodes.
                List<Integer> validNodesList = new ArrayList<>();
                if (isFrozen[i]) {
                    // Frozen: confined to nodes it's ALREADY resident on -- no new node can ever
                    // be reached (transferHeights fixed to false there above), so letting jobNodes
                    // range over the rest would only let the solver explore placements it will
                    // then find infeasible. Tasks can still move among its own existing replicas.
                    for (int n : replicas_location[i]) validNodesList.add(n);
                } else {
                    for (int j = 0; j < nb_nodes; j++) {
                        if (data_sizes[i] <= storage_capacity[j]) validNodesList.add(j);
                    }
                }
                int[] validNodes = validNodesList.isEmpty()
                        ? ArrayUtils.array(0, nb_nodes - 1) // instance infaisable ; on laisse les autres contraintes le detecter
                        : validNodesList.stream().mapToInt(Integer::intValue).toArray();


                //System.out.println("Creating work tasks for data " + i + " with " + wl.length + " works.");
                for (int k = 0; k < wl.length; k++) {
                    int w = wl[k];
                    jobStarts[i][k] = model.intVar("start_work_d" + i + "_w" + k, 0, makespan,true); //(int) starting_times[j]
                    int[] durations = new int[nb_nodes];
                    for (int j = 0; j < nb_nodes; j++) {
                        durations[j] = (int) (w * cpus[j]);
                    }
                    int min = Arrays.stream(durations).min().getAsInt();
                    int max = Arrays.stream(durations).max().getAsInt();
                    jobDurations[i][k] = model.intVar("duration_work_d" + i + "_w" + k, min, max);
                    jobNodes[i][k] = model.intVar("node_work_d" + i + "_w" + k, validNodes);
                    model.element(jobDurations[i][k], durations, jobNodes[i][k]).post();
                    jobEnds[i][k] = model.intVar("end_work_d" + i + "_w" + k, 0, makespan,true);//(int) starting_times[j]
                    model.arithm(jobStarts[i][k], "+", jobDurations[i][k], "=", jobEnds[i][k]).post();
                    for (int j = 0; j < nb_nodes; j++) {
                        BoolVar jOnN = jobNodes[i][k].eq(j).boolVar();
                        // A work can start only after the corresponding transfer is finished on that node,
                        model.impXrelYC(jobStarts[i][k], ">=", transferTasks[j][i].getEnd(), 0, jOnN);
                    }
                    //System.out.println("Duration variable for work " + k + " of data " + i + ": " + jobDurations[i][k]);
                }
            }



            //  if a transfer happens, then at least one work must happen on that node for that data
            IntVar[][] nb_transfers = new IntVar[nb_data][nb_nodes];
            for (int i = 0; i < nb_data; i++) {
                for (int j = 0; j < nb_nodes; j++) {
                    nb_transfers[i][j] = transferHeights[j][i].intVar();
                }
            }
            int factor = 1;
            for (int i = 0; i < nb_data; i++) {
                int[] wl = works[i];
                IntVar[] counters = new IntVar[nb_nodes];
                for (int j = 0; j < nb_nodes; j++) {
                    counters[j] = model.intVar(0, wl.length);
                    //model.count(j, jobNodes[i], counters[j]).post();
                    model.reifXrelC(counters[j], ">=", 1, transferHeights[j][i]);
                    // constraintes redondantes
                    for (int k = 0; k < wl.length - 1; k++) {
                        transferHeights[j][i].eq(0).imp(jobNodes[i][k].ne(j)).post();
                    }
                }
                model.globalCardinality(jobNodes[i], ArrayUtils.array(0, nb_nodes - 1), counters, true).post();
                model.sum(counters, "=", wl.length).post();
                // Transfer_time <= factor * sum(execution_time)
                for (int j = 0; j < nb_nodes && factor > 0; j++) {
                    IntVar[] executions = new IntVar[works[i].length];
                    for (int k = 0; k < executions.length; k++) {
                        executions[k] = model.isEq(jobNodes[i][k], j).mul(jobDurations[i][k]).intVar();
                    }
                    IntVar transfers_counter = model.intVar(0, nb_nodes);
                    model.sum(nb_transfers[i], "=", transfers_counter).post();
                    BoolVar h = transfers_counter.gt(1).and(transferHeights[j][i]).boolVar();
                    int transferTime = (int) Math.ceil((double) data_sizes[i] / bandwidths[j]);
                    model.sum(executions, ">=", model.intView(transferTime * factor, h, 0)).post();
                    //counters[j].gt(1).imp(exe).post();
                }

                for (int k = 0; k < wl.length - 1; k++) {
                    jobNodes[i][k].eq(jobNodes[i][k + 1]).imp(jobEnds[i][k].eq(jobStarts[i][k + 1])).post();
                    // very strict :
                    jobNodes[i][k].le(jobNodes[i][k + 1]).post();
                }
            }


            //// ----- STORAGE CONSTRAINT -----
            //// Data i occupies storage on node j from its effective start until the last work
            //// assigned to that node for that data finishes (release time) -- that's when it can
            //// be deleted locally. No separate keep-vs-abandon decision variable: if the solver
            //// never routes a task for data i to node j, transferHeights[j][i] is forced to 0 (via
            //// the counters/reification link below), storageHeights[j][i] collapses to 0, and
            //// release falls back to effectiveStart -- i.e. the model itself already treats
            //// "nothing uses it here" as "abandon it now", with nothing extra to branch on.
            //Task[][] storageTasks = new Task[nb_nodes][nb_data];
            //IntVar[][] storageHeights = new IntVar[nb_nodes][nb_data];
            //IntVar[][] releases = new IntVar[nb_nodes][nb_data];
            //boolean[][] alreadyResident = new boolean[nb_nodes][nb_data];
            //for (int i = 0; i < nb_data; i++) {
            //    int[] wl = works[i];
            //    for (int n : replicas_location[i]) {
            //        if (n >= 0 && n < nb_nodes) alreadyResident[n][i] = true;
            //    }
            //    for (int j = 0; j < nb_nodes; j++) {
            //        // if a data is already present on a node, its effective start is 0, not the transfer start
            //        IntVar effectiveStart = alreadyResident[j][i]
            //                ? model.intVar(0)
            //                : transferTasks[j][i].getStart();
            //        // release_ij = max end time among works of data i actually assigned to node j;
            //        // falls back to effectiveStart when no work of i is on j (storage then
            //        // irrelevant since storageHeights[j][i] will be 0 in that case).
            //        IntVar[] candidateEnds = new IntVar[wl.length];
            //        for (int k = 0; k < wl.length; k++) {
            //            BoolVar onJ = jobNodes[i][k].eq(j).boolVar();
            //            IntVar cand = model.intVar("release_cand_d" + i + "_n" + j + "_w" + k, 0, makespan, true);
            //            model.impXrelYC(cand, "=", jobEnds[i][k], 0, onJ);
            //            model.impXrelYC(cand, "=", effectiveStart, 0, onJ.not());
            //            candidateEnds[k] = cand;
            //        }
            //        IntVar release = model.intVar("release_d" + i + "_n" + j, 0, makespan, true);
            //        model.max(release, candidateEnds).post();
            //        releases[j][i] = release;
            //        IntVar storageDuration = model.intVar("storage_duration_d" + i + "_n" + j, 0, makespan, true);
            //        storageTasks[j][i] = new Task(effectiveStart, storageDuration, release);
            //        storageHeights[j][i] = transferHeights[j][i].mul(data_sizes[i]).intVar();
            //    }
            //}
            //for (int j = 0; j < nb_nodes; j++) {
            //    List<Task> nodeStorageTasks = new ArrayList<>(Arrays.asList(storageTasks[j]));
            //    List<IntVar> nodeStorageHeights = new ArrayList<>(Arrays.asList(storageHeights[j]));
            //    for (GhostStorage g : ghostStorage) {
            //        if (g.nodeId != j) continue;
            //        // Fixed (non-decision) task: occupies g.size from 0 until its known/assumed
            //        // release time, exactly like alreadyResident data, just not schedulable here.
            //        Task ghostTask = new Task(model.intVar(0), model.intVar(g.deletionTime), model.intVar(g.deletionTime));
            //        nodeStorageTasks.add(ghostTask);
            //        nodeStorageHeights.add(model.intVar(g.size));
            //    }
            //    model.cumulative(
            //            nodeStorageTasks.toArray(new Task[0]),
            //            nodeStorageHeights.toArray(new IntVar[0]),
            //            model.intVar(storage_capacity[j])
            //    ).post();
            //
            //}

            // ----- STORAGE CONSTRAINT -----
            // Data i occupies storage on node j from the moment its transfer to j
            // starts until the last work assigned to that node for that data
            // finishes (release time) -- that's when it can be deleted locally.
            Task[][] storageTasks = new Task[nb_nodes][nb_data];
            IntVar[][] storageHeights = new IntVar[nb_nodes][nb_data];
            for (int i = 0; i < nb_data; i++) {
                int[] wl = works[i];
                for (int j = 0; j < nb_nodes; j++) {
                    IntVar transferStart = transferTasks[j][i].getStart();

                    // release_ij = max end time among works of data i actually assigned to node j;
                    // falls back to transferStart when no work of i is on j (storage then irrelevant
                    // since storageHeights[j][i] will be 0).
                    IntVar[] candidateEnds = new IntVar[wl.length];
                    for (int k = 0; k < wl.length; k++) {
                        BoolVar onJ = jobNodes[i][k].eq(j).boolVar();
                        IntVar cand = model.intVar("release_cand_d" + i + "_n" + j + "_w" + k, (int)starting_times[j], makespan, true);
                        model.impXrelYC(cand, "=", jobEnds[i][k], 0, onJ);
                        model.impXrelYC(cand, "=", transferStart, 0, onJ.not());
                        candidateEnds[k] = cand;
                    }
                    IntVar release = model.intVar("release_d" + i + "_n" + j, (int)starting_times[j], makespan, true);
                    model.max(release, candidateEnds).post();

                    IntVar storageDuration = model.intVar("storage_duration_d" + i + "_n" + j, 0, makespan, true);
                    storageTasks[j][i] = new Task(transferStart, storageDuration, release);
                    storageHeights[j][i] = transferHeights[j][i].mul(data_sizes[i]).intVar();
                }
            }
            for (int j = 0; j < nb_nodes; j++) {
                model.cumulative(storageTasks[j], storageHeights[j], model.intVar(storage_capacity[j])).post();
            }

            //for (int j = 0; j < nb_nodes; j++) {
            //    List<Task> nodeStorageTasks = new ArrayList<>(Arrays.asList(storageTasks[j]));
            //    List<IntVar> nodeStorageHeights = new ArrayList<>(Arrays.asList(storageHeights[j]));
            //    for (GhostStorage g : ghostStorage) {
            //        if (g.nodeId != j) continue;
            //        // Fixed (non-decision) task: occupies g.size from 0 until its known/assumed
            //        // release time, exactly like alreadyResident data, just not schedulable here.
            //        Task ghostTask = new Task(model.intVar(0), model.intVar(g.deletionTime), model.intVar(g.deletionTime));
            //        nodeStorageTasks.add(ghostTask);
            //        nodeStorageHeights.add(model.intVar(g.size));
            //    }
            //    model.cumulative(
            //            nodeStorageTasks.toArray(new Task[0]),
            //            nodeStorageHeights.toArray(new IntVar[0]),
            //            model.intVar(storage_capacity[j])
            //    ).post();
            //}

            // ----- CONSTRAINTS -----
            // Cumulative constraints for transfers on each node (capacity = 1)
            for (int j = 0; j < nb_nodes; j++) {
                Task[] tasksForNode = new Task[nb_data];
                IntVar[] heightsForNode = new IntVar[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    tasksForNode[i] = transferTasks[j][i];
                    heightsForNode[i] = transferHeights[j][i];
                }
                // capacity = 1
                // System.out.printf("Task_%d -- duration= %s%n", j, tasksForNode[0].getDuration());
                model.cumulative(tasksForNode, heightsForNode, model.intVar(1)).post();
            }
            // At least one transfer per data (sum over nodes heights[j][i] >= 1)
            for (int i = 0; i < nb_data; i++) {
                IntVar[] arr = new IntVar[nb_nodes];
                for (int j = 0; j < nb_nodes; j++) arr[j] = transferHeights[j][i];
                model.sum(arr, ">=", 1).post();
            }

            // Each work must be done exactly once (sum of heights for a given (data i, work k) across nodes == 1)
            model.diffN(ArrayUtils.flatten(jobStarts),
                    ArrayUtils.flatten(jobNodes),
                    ArrayUtils.flatten(jobDurations),
                    Arrays.stream(ArrayUtils.flatten(jobNodes)).map(j -> model.intVar(1)).toArray(IntVar[]::new),
                    true).post();

            // ----- OBJECTIVE Make span-----
            //makespan var and ensure it's >= all end
            boolean makespan_obj = false;
            final IntVar[] objectives = new IntVar[4];
            if (makespan_obj) {
                /*IntVar makespanVar = model.intVar("makespan", 0, makespan);

                
                model.setObjective(false, makespanVar); // false => MINIMIZE (see Choco API)*/
            } else {

                IntVar[] all_flow_time = new IntVar[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    IntVar end_time = model.intVar(0, makespan);
                    model.max(end_time, jobEnds[i]).post();

                    IntVar elapsedTime =  model.intVar(jobs.get(i).timelasped);

                    // flow = end_time + elapsedTime can exceed makespan by however long this job
                    // already waited across earlier replans, so its domain must be wider than
                    // end_time's own -- capping it at plain makespan would make the model
                    // spuriously infeasible for any already-long-waiting job (same class of bug
                    // fixed for MainOnlineThreeStep.java's storage "release" bound).
                    IntVar flow = model.intVar(0, 999_999);
                    model.arithm(end_time, "+", elapsedTime, "=", flow).post();

                    all_flow_time[i] = flow;
                }
                IntVar maxFlowTime = model.intVar("max_flow_time", 0, 999_999);
                model.max(maxFlowTime, all_flow_time).post();
                IntVar sumFlowTime = model.intVar("sum_flow_time", 0, 999_999);
                model.sum(all_flow_time, "=", sumFlowTime).post();
                //model.setObjective(false, maxFlowTime);
                objectives[1] = maxFlowTime;
                objectives[0] = sumFlowTime;

                // Objective 2: the flow time of ONE specific job -- by convention, whichever job
                // in this batch has the HIGHEST job_id. Every experiment driving this from Python
                // gives the brand-new job an id far above any existing job's (see
                // xp_online_warmstart_test.py's NEW_JOB_ID_BASE), so after modelCSP.py's own
                // sort-by-job_id this is always that new job -- letting Online be told to
                // optimize purely for the new arrival's own flow time, instead of the whole
                // batch's sum/max.
                int newJobIdx = 0;
                for (int i = 1; i < nb_data; i++) {
                    if (jobs.get(i).job_id > jobs.get(newJobIdx).job_id) newJobIdx = i;
                }
                objectives[2] = all_flow_time[newJobIdx];

                // Objective 3: total transfer energy (sender + receiver*duration + network,
                // summed over every SELECTED (node,data) candidate) -- see energyContrib above.
                // Linear in the transferHeights BoolVars since each candidate's own cost is a
                // known constant once (i,j) is fixed; only which candidates get chosen is a
                // decision. Upper-bounded by the (unreachable in practice) sum of EVERY
                // candidate's cost, since selecting fewer can only reduce the total.
                long energyUpperBoundLong = 0;
                for (int[] row : energyContrib) for (int v : row) energyUpperBoundLong += v;
                int energyUpperBound = (int) Math.min(energyUpperBoundLong, Integer.MAX_VALUE / 2);
                IntVar energyVar = model.intVar("total_energy", 0, energyUpperBound, true);
                IntVar[] transferHeightsFlat = new IntVar[nb_nodes * nb_data];
                int[] energyCoeffsFlat = new int[nb_nodes * nb_data];
                int ecIdx = 0;
                for (int j = 0; j < nb_nodes; j++) {
                    for (int i = 0; i < nb_data; i++) {
                        transferHeightsFlat[ecIdx] = transferHeights[j][i];
                        energyCoeffsFlat[ecIdx] = energyContrib[j][i];
                        ecIdx++;
                    }
                }
                model.scalar(transferHeightsFlat, energyCoeffsFlat, "=", energyVar).post();
                objectives[3] = energyVar;
            }

            //----- SOLVER -----
            Solver solver = model.getSolver();
            List<TransferConfig> transfersList = new ArrayList<>();
            List<WorkConfig> worksList = new ArrayList<>();
            List<DeletionConfig> deletionsList = new ArrayList<>();
            //model.displayVariableOccurrences();
            //model.displayPropagatorOccurrences();

            IntVar[] decisionVars = decisionVariables(nb_nodes, nb_data, works, jobNodes, jobStarts, transferHeights, transferTasks);
            // TEMP: hints disabled to check whether they're locking in the job11-style idle gaps
            // hints(nb_nodes, nb_data, data_sizes, works, cpus, solver, jobNodes);

            //solver.setNoGoodRecordingFromRestarts();
            ArraySort<?> sorter = new ArraySort<>(nb_nodes, false, true);
            int[] cidx = ArrayUtils.array(0, nb_nodes - 1);
            sorter.sort(cidx, nb_nodes, (i, j) -> (int) ((cpus[i] - cpus[j]) * 1000));
            solver.setSearch(
                    Search.lastConflict(
                            Search.intVarSearch(new InputOrder<>(model),
                                    new IntDomainLast(model.getSolver().defaultSolution(),
                                            new IntValueSelector() {
                                                @Override
                                                public int selectValue(IntVar intVar) {
                                                    if (intVar.getName().startsWith("node_work_d")) {
                                                        int i = 0;
                                                        while (i < nb_nodes) {
                                                            if (intVar.contains(cidx[i])) {
                                                                return cidx[i];
                                                            }
                                                            i++;
                                                        }
                                                    }
                                                    return intVar.getLB();
                                                }
                                            }, (i, j) -> true),
                                    decisionVars), 2)
            );
            if (solve || pos == 1) {
                setLNS(solver,
                        //new AdaptiveNeighborhood(42,
                        new SequenceNeighborhood(
                                getNeighbor1(nb_data, works, jobNodes, jobStarts, jobEnds),
                                getNeighbor2(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor3(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor1bis(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor2bis(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor3bis(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor4bis(nb_data, works,  jobNodes, jobStarts, jobEnds)
                        ),
                        new FailCounter(model, nb_data * nb_nodes * 100));
            } else if (pos == 2) {
                setLNS(solver,
                        new SequenceNeighborhood(
                                getNeighbor1bis(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor3bis(nb_data, works,  jobNodes, jobStarts, jobEnds)
                        ),
                        new FailCounter(model, nb_data * nb_nodes * 50));
            } else if (pos == 3) {
                setLNS(solver,
                        new SequenceNeighborhood(
                                getNeighbor1(nb_data, works, jobNodes, jobStarts, jobEnds),
                                getNeighbor2bis(nb_data, works,  jobNodes, jobStarts, jobEnds),
                                getNeighbor3bis(nb_data, works,  jobNodes, jobStarts, jobEnds)
                        ),
                        new FailCounter(model, nb_data * nb_nodes * 100));
            }

            int timeLimitSeconds = 30;
            try {
                String timeLimitText = readFile(MODEL_INPUTS_DIR + "/solver_time_limit.txt").trim();
                timeLimitSeconds = Integer.parseInt(timeLimitText);
            } catch (Exception e) {
                // File missing/unreadable (e.g. an older Python caller that doesn't write it
                // yet): keep the 120s default rather than fail the whole solve over this.
            }
            System.out.println("Solver time limit: " + timeLimitSeconds + "s");
            solver.limitTime(timeLimitSeconds + "s");

            // Which objectives[] entry to actually optimize: 0=sum flow time (all jobs), 1=max
            // flow time (all jobs, the long-standing default), 2=the new job's own flow time
            // alone (see objectives[2] above). Runtime-selectable so every OTHER experiment that
            // never writes this file keeps today's exact behavior (default: 1).
            int objectiveChoice = 1;
            try {
                String objectiveChoiceText = readFile(MODEL_INPUTS_DIR + "/objective_choice.txt").trim();
                if (!objectiveChoiceText.isEmpty()) objectiveChoice = Integer.parseInt(objectiveChoiceText);
            } catch (Exception e) {
                // File missing/unreadable: keep the default (1 = max flow time, all jobs).
            }
            System.out.println("Objective choice: " + objectiveChoice
                    + " (0=sum all, 1=max all, 2=new job's own flow time)");

            boolean[] found = {false};
            // Captured INSIDE onSolution (where every variable is guaranteed instantiated, since
            // a solution was just accepted) rather than read from objectives[1] AFTER
            // findOptimalSolution returns -- when the time limit cuts the search off mid-branch
            // (not via exhaustive proof of optimality), the model's live variable state reflects
            // wherever the search currently is, NOT necessarily the last accepted solution, and
            // can be a not-yet-fully-instantiated intermediate node. Reading objectives[1]
            // .getValue() directly in that case throws IllegalStateException (seen in practice
            // on some -- not all -- runs, since it depends on the exact search state when the
            // clock runs out). epsilon-constraint mode (below) needs phase 1's best value to set
            // up phase 2's cap, so it must use this holder, not objectives[1] directly.
            int[] lastMaxFlow = {-1};
            int[] lastEnergy = {-1};
            // Set right before phase 2 starts (epsilon-constraint mode only) so onSolution can
            // report, on phase 2's very FIRST accepted solution, whether the warm start actually
            // took (i.e. that first solution's maxFlowTime should equal phase 1's own, since the
            // search is expected to re-derive phase 1's exact assignment before improving energy
            // any further -- see the warm-start comment where phase 2 is set up below).
            boolean[] inPhase2 = {false};
            boolean[] phase2FirstSolutionSeen = {false};
            int[] phase1FinalMaxFlow = {-1};
            solver.onSolution(() -> {

                    System.out.println("### DIAG solution found: sumFlowTime=" + objectives[0].getValue()
                            + " maxFlowTime=" + objectives[1].getValue()
                            + " newJobFlowTime=" + objectives[2].getValue()
                            + " energy=" + objectives[3].getValue() + " nb_data=" + nb_data + " t=" + solver.getTimeCount());
                    found[0] = true;
                    lastMaxFlow[0] = objectives[1].getValue();
                    if (inPhase2[0] && !phase2FirstSolutionSeen[0]) {
                        phase2FirstSolutionSeen[0] = true;
                        boolean warmStartHeld = lastMaxFlow[0] == phase1FinalMaxFlow[0];
                        System.out.println("### EPSILON-CONSTRAINT: phase2's FIRST solution has "
                                + "maxFlowTime=" + lastMaxFlow[0] + " (phase1's was " + phase1FinalMaxFlow[0]
                                + ") -- warm start " + (warmStartHeld ? "HELD (search re-derived phase 1's "
                                + "exact solution before improving energy)" : "DID NOT hold exactly (search "
                                + "found a different, still cap-respecting point first instead)") + " ###");
                    }
                    lastEnergy[0] = objectives[3].getValue();

                    transfersList.clear();
                    worksList.clear();
                    deletionsList.clear();

                    for (int j = 0; j < nb_nodes; j++) {

                        for (int i = 0; i < nb_data; i++) {
                            if (transferHeights[j][i].getValue() == 1) {

                                int[] wl = works[i];
                                for (int k = 0; k < wl.length; k++) {
                                    if (jobNodes[i][k].isInstantiatedTo(j)) {

                                        WorkConfig tmp_work = new WorkConfig(k, i, jobStarts[i][k].getValue(), jobEnds[i][k].getValue(), j);
                                        worksList.add(tmp_work);
                                    }
                                }

                                TransferConfig tmp_transfer = new TransferConfig(i, transferTasks[j][i].getStart().getValue(), transferTasks[j][i].getEnd().getValue(), j);
                                transfersList.add(tmp_transfer);

                                // Freshly-transferred data's own release time (computed by the
                                // storage cumulative constraint above, storageTasks[j][i]'s end)
                                // was never exported before -- only abandon decisions for
                                // already-resident replicas were. Without this, nothing downstream
                                // (the simulator's own bookkeeping, any storage-occupancy
                                // reconstruction) can tell when this node frees back up, making
                                // freshly-placed data look permanently resident.
                                deletionsList.add(new DeletionConfig(i, j, storageTasks[j][i].getEnd().getValue()));
                            }

                            final int jFinal = j;
                            if(transferHeights[j][i].getValue() == 0 && replicas_location[i].length > 0 && Arrays.stream(replicas_location[i]).anyMatch(n -> n == jFinal)) {
                                // Not 0 (immediate): nodeStartingTimes[j] already reflects when
                                // node j's CURRENTLY ongoing transfer/task (whoever it belongs
                                // to) actually finishes. Deleting any earlier risks yanking data
                                // out from under a task that's mid-execution but no longer
                                // visible to this solve (already-Started tasks aren't part of
                                // the work list, so an abandon decision here has no way to know
                                // about them otherwise).
                                deletionsList.add(new DeletionConfig(i, j, nodeStartingTimes[j]));
                            }


                            // Storage on (j,i) is occupied within this solve either because it's
                            // used now (height=1, release = when the last task using it there
                            // finishes) or because it was already resident before this solve but
                            // is abandoned here (height=0, so release collapses to effectiveStart
                            // -- i.e. delete it right away since nothing here needs it anymore).
                            //if (transferHeights[j][i].getValue() == 1 || alreadyResident[j][i]) {
                            //    // int releaseTime = releases[j][i].getValue();
                            //    if (releaseTime < makespan) {
                            //        deletionsList.add(new DeletionConfig(i, j, 0)); // releaseTime));
                            //    }
                            //}
                        }
                    }

                /*System.out.printf("%d;%d;%.2f;%d\n",
                        objectives[0].getValue(), objectives[1].getValue(), solver.getTimeCount(), solver.getSolutionCount());*/
            });

            System.out.printf("Node free/release times before solving (sim clock now=%.4f):%n", currentSimTime);
            for (int j = 0; j < nb_nodes; j++) {
                System.out.printf("  node_%d: raw=%.4f -> used=%d  (absolute: now+used=%.4f)%n",
                        j, starting_times[j], nodeStartingTimes[j], currentSimTime + nodeStartingTimes[j]);
            }

            // Multi-objective mode, read from multi_objective.txt: 0/absent = ordinary
            // single-objective findOptimalSolution (unchanged); 1 = raw Pareto-front search over
            // {max flow time, energy} via ParetoMaximizer -- empirically found to perform far
            // WORSE than single-objective search within the same budget on real scenarios (the
            // existing LNS/restart search strategy is tuned to aggressively descend ONE scalar
            // objective, and doesn't explore a 2D dominance frontier well: on a 5-job real
            // scenario it got stuck oscillating between 2 points, both much worse than
            // single-objective's result, despite exploring 200k+ solutions in 600s); 2 =
            // epsilon-constraint, a two-phase approach that reuses the SAME well-tuned
            // single-objective search machinery for both phases instead: phase 1 minimizes max
            // flow time as usual, phase 2 then minimizes energy subject to max flow time staying
            // within an epsilon slack of phase 1's result -- see the mode==2 branch below.
            int multiObjectiveMode = 0;
            try {
                String multiObjectiveText = readFile(MODEL_INPUTS_DIR + "/multi_objective.txt").trim();
                if (!multiObjectiveText.isEmpty()) multiObjectiveMode = Integer.parseInt(multiObjectiveText);
            } catch (Exception e) {
                // File missing/unreadable: keep the default (0 = single-objective search).
            }

            if (multiObjectiveMode == 1) {
                System.out.println("### Multi-objective mode: searching for the Pareto front over "
                        + "{max flow time, energy} ###");
                // ParetoMaximizer only MAXIMIZES -- model.neg(...) views let it maximize the
                // negated vars, equivalent to minimizing the originals, without adding real
                // constraints or duplicating the actual objective IntVars.
                ParetoMaximizer pareto = new ParetoMaximizer(new IntVar[]{model.neg(objectives[1]), model.neg(objectives[3])});
                model.post(new Constraint("PARETO_MAXFLOW_ENERGY", pareto));
                // Posting it as a constraint only wires its PROPAGATION (pruning dominated
                // points during search) -- its own onSolution() bookkeeping (what actually
                // fills getParetoFront()) is a SEPARATE ISearchMonitor hook that must be plugged
                // explicitly, or the front comes back empty even though solutions were found.
                solver.plugMonitor(pareto);
                while (solver.solve()) { /* pareto's own onSolution() records each non-dominated point */ }
                if (!found[0]) {
                    System.out.println("No solution found");
                }
                List<Solution> paretoFront = pareto.getParetoFront();
                System.out.println("### PARETO FRONT: " + paretoFront.size() + " solution(s) ###");
                for (Solution sol : paretoFront) {
                    System.out.println("  maxFlowTime=" + sol.getIntVal(objectives[1])
                            + "  energy=" + sol.getIntVal(objectives[3]));
                }
            } else if (multiObjectiveMode == 2) {
                // epsilon_fraction: how much worse than phase 1's own best max-flow-time result
                // phase 2 is allowed to make it, as a FRACTION of that result (not an absolute
                // value -- flow times vary wildly in magnitude across scenarios). Default 10%.
                double epsilonFraction = 0.1;
                try {
                    String t = readFile(MODEL_INPUTS_DIR + "/epsilon_fraction.txt").trim();
                    if (!t.isEmpty()) epsilonFraction = Double.parseDouble(t);
                } catch (Exception e) { /* default 0.1 */ }
                // How the total solver_time_limit.txt budget splits between the two phases.
                // Default even split; phase 1 usually doesn't need as long as phase 2 in
                // practice (it's a strictly easier, already well-tuned single-objective search),
                // but an even split is a safe, simple default.
                double phase1Fraction = 0.5;
                try {
                    String t = readFile(MODEL_INPUTS_DIR + "/epsilon_phase1_fraction.txt").trim();
                    if (!t.isEmpty()) phase1Fraction = Double.parseDouble(t);
                } catch (Exception e) { /* default 0.5 */ }
                // Optional absolute ceiling on phase 2's cap -- e.g. a baseline (Incremental's)
                // own max flow time result from a prior run, so this approach never needs to
                // accept worse flow time than that trivial baseline already gets for free just
                // because epsilonFraction's relative slack happened to push past it (which
                // otherwise silently gets more likely the less phase 1's own budget lets it
                // converge -- see the large-tier run where a 1h/1h split let the relative cap
                // drift to +18.5% over Online's own 2h result). Absent -> no ceiling, unchanged
                // behavior.
                Double epsilonMaxCap = null;
                try {
                    String t = readFile(MODEL_INPUTS_DIR + "/epsilon_max_cap.txt").trim();
                    if (!t.isEmpty()) epsilonMaxCap = Double.parseDouble(t);
                } catch (Exception e) { /* default: no ceiling */ }

                int phase1Seconds = Math.max(1, (int) Math.round(timeLimitSeconds * phase1Fraction));
                int phase2Seconds = Math.max(1, timeLimitSeconds - phase1Seconds);
                System.out.println("### EPSILON-CONSTRAINT mode: phase1 (minimize max flow time) budget="
                        + phase1Seconds + "s, phase2 (minimize energy under flow-time cap) budget="
                        + phase2Seconds + "s, epsilon fraction=" + epsilonFraction + " ###");

                solver.limitTime(phase1Seconds + "s");
                solver.findOptimalSolution(objectives[1], false);
                if (!found[0]) {
                    System.out.println("No solution found (phase 1)");
                } else {
                    // From the holder (captured inside onSolution, always fully instantiated
                    // there), NOT objectives[1].getValue() directly -- if the time limit cut the
                    // search off mid-branch rather than via an exhaustive optimality proof, the
                    // model's LIVE variable state reflects wherever the search currently sits,
                    // which is not necessarily the last accepted solution and can throw
                    // IllegalStateException ("not instantiated") depending on the exact search
                    // state when the clock ran out.
                    int bestMaxFlow = lastMaxFlow[0];
                    int epsilonAbs = Math.max(1, (int) Math.round(bestMaxFlow * epsilonFraction));
                    int cap = bestMaxFlow + epsilonAbs;
                    if (epsilonMaxCap != null) {
                        // Never below bestMaxFlow itself: phase 1 already PROVED that value is
                        // achievable, so a ceiling under it would make phase 2's own constraint
                        // infeasible from the start. If the ceiling is that tight, phase 2 simply
                        // gets zero slack (cap == bestMaxFlow) instead of crashing.
                        int ceilingInt = (int) Math.round(epsilonMaxCap);
                        int cappedCap = Math.max(bestMaxFlow, Math.min(cap, ceilingInt));
                        if (cappedCap != cap) {
                            System.out.println("### EPSILON-CONSTRAINT: relative cap " + cap
                                    + " exceeds the absolute ceiling " + ceilingInt
                                    + " -- clamping phase2's cap to " + cappedCap + " ###");
                        }
                        cap = cappedCap;
                    }
                    System.out.println("### EPSILON-CONSTRAINT: phase1 best maxFlowTime=" + bestMaxFlow
                            + "  phase2 cap=maxFlowTime<=" + cap + " (epsilon=" + epsilonAbs + ") ###");

                    // reset() clears the search tree, measures (getTimeCount() back to 0), and
                    // every previously-set stop criterion (including phase 1's limitTime) --
                    // WITHOUT undoing already-posted constraints or the model's own propagated
                    // domains, so the newly-added cap constraint below stacks cleanly on top of
                    // everything phase 1 already established, and phase 2's own limitTime call
                    // counts fresh from this point rather than cumulatively from phase 1's.
                    //
                    // WARM START: phase 2 must not re-explore from scratch -- it should start
                    // from phase 1's own solution, which trivially still satisfies the new cap
                    // (cap = phase1's maxFlowTime + epsilon >= phase1's maxFlowTime). The search
                    // strategy set up earlier in this method already branches via
                    // IntDomainLast(solver.defaultSolution(), ...) -- Solver.defaultSolution()
                    // is a Solution object Choco auto-attaches on first use and keeps updated on
                    // EVERY accepted solution (safe to read even after a time-limit cutoff,
                    // unlike a live IntVar's .getValue()), and reset() does NOT clear or detach
                    // it (confirmed against Choco 5's own Solver.reset() source: it resets the
                    // search tree/measures/stop criteria, never the attached solution recorder).
                    // So by the time phase 2 starts branching, defaultSolution() already holds
                    // phase 1's final assignment for every decision variable, and the SAME value
                    // selector (unchanged -- setSearch() is only ever called once, before phase
                    // 1) will try to reconstruct exactly that assignment first, before searching
                    // for anything better. The onSolution callback above verifies this actually
                    // happens (phase 2's first accepted solution's maxFlowTime is checked against
                    // phase1FinalMaxFlow) rather than just assuming it from Choco's internals.
                    phase1FinalMaxFlow[0] = bestMaxFlow;
                    inPhase2[0] = true;
                    solver.reset();
                    model.arithm(objectives[1], "<=", cap).post();
                    solver.limitTime(phase2Seconds + "s");
                    found[0] = false; // phase 2's own outcome, tracked separately from phase 1's
                    solver.findOptimalSolution(objectives[3], false);
                    if (!found[0]) {
                        System.out.println("### EPSILON-CONSTRAINT: no solution found in phase 2 -- "
                                + "the exported schedule is whatever phase 1 last left in "
                                + "transfersList/worksList (maxFlowTime=" + bestMaxFlow + ") ###");
                    } else {
                        System.out.println("### EPSILON-CONSTRAINT: phase2 best energy=" + lastEnergy[0]
                                + "  (maxFlowTime=" + lastMaxFlow[0] + ") ###");
                    }
                }
            } else {
                solver.findOptimalSolution(objectives[objectiveChoice], false);
                if (!found[0]) {
                    System.out.println("No solution found");
                }
            }
            System.out.println("### DIAG search end: timeCount=" + solver.getTimeCount()
                    + "s  timeLimitWas=" + timeLimitSeconds + "s  objectiveOptimal=" + solver.isObjectiveOptimal()
                    + "  solutionCount=" + solver.getSolutionCount());

            SchedulingResult result = new SchedulingResult(transfersList, worksList, deletionsList);

            return result;
        }

        private static IntVar[] decisionVariables(int nb_nodes, int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, BoolVar[][] transferHeights, Task[][] transferTasks) {
            List<IntVar> vars = new ArrayList<>();
            for (int i = 0; i < nb_data; i++) {
                for (int k = 0; k < works[i].length; k++) {
                    vars.add(jobNodes[i][k]);
                    vars.add(jobStarts[i][k]);
                }
            }
            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {
                    vars.add(transferHeights[j][i]);
                    vars.add(transferTasks[j][i].getStart());
                }
            }
            IntVar[] decisionVars = vars.toArray(new IntVar[0]);
            return decisionVars;
        }

        private static void hints(int nb_nodes, int nb_data, int[] data_sizes, int[][] works, double[] cpus, Solver solver, IntVar[][] jobNodes) {
            ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);
            int[] didx = ArrayUtils.array(0, nb_data - 1);
            sorter.sort(didx, nb_data, (i, j) -> {
                int diff = data_sizes[j] - data_sizes[i];
                if (diff == 0) {
                    diff = works[j].length - works[i].length;
                }
                return diff;
            });
            sorter = new ArraySort<>(nb_nodes, false, true);
            int[] cidx = ArrayUtils.array(0, nb_nodes - 1);
            sorter.sort(cidx, nb_nodes, (i, j) -> (int) ((cpus[i] - cpus[j]) * 1000));
            for (int i = 0; i < nb_data; i++) {
                int k = 0;
                // cidx only has nb_nodes entries: cycle through them once there are more
                // data items than nodes (pre-existing bug, previously untriggered because
                // every shipped instance had nb_nodes >= nb_data).
                int ii = cidx[i % nb_nodes];
                for (; k < works[i].length; k++) {
                    solver.addHint(jobNodes[i][k], ii);
                }
            }
        }

        public static void setLNS(Solver solver, INeighbor neighbor, ICounter restartCounter) {
            MyMoveLNS lns = new MyMoveLNS(solver.getMove(), neighbor, restartCounter);
            solver.setMove(lns);
        }

        private static INeighbor getNeighbor0(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                @Override
                public void recordSolution() {
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                }

                @Override
                public void loadFromSolution(Solution solution) {
                }

                @Override
                public void restrictLess() {
                }
            };
        }

        private static INeighbor getNeighbor1(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int lim = 0;
                int loops = 0;
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    lim = 0;
                    loops = 1;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    for (int i = 0; i < nb_data /*&& loops < 1000*/; i++) {
                        int ii = imaxs[i];
                        for (int k = 0; k < works[ii].length; k++) {
                            if (i == lim) {
                                jobNodes[ii][k].removeValue(jn[ii][k], this);
                            } else {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                    }
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                    lim = (lim + 1) % nb_data;
                    if (lim == 0) {
                        loops++;
                        //System.out.printf("Loops %d\n", loops);
                    }
                }
            };
        }

        private static INeighbor getNeighbor2(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int data = 0;
                int work = 0;
                final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    mapping.clear();
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            TIntArrayList list = mapping.get(i);
                            if (list == null) {
                                list = new TIntArrayList();
                                mapping.put(i, list);
                            }
                            if (!list.contains(jn[i][k])) {
                                list.add(jn[i][k]);
                            }
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    data = 0;
                    work = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    boolean move = false;
                    for (int i = 0; i < nb_data; i++) {
                        int ii = imaxs[i];
                        TIntArrayList values = mapping.get(ii);
                        if (i == data) {
                            // Choco can backtrack past an earlier call to this method (a
                            // contradiction found elsewhere undoes the variable instantiations
                            // made here), but data/work/mapping/jn/sn are plain Java fields --
                            // not part of Choco's trail -- so they are NOT rolled back with it.
                            // On retry, `work` can end up pointing past the end of this job's
                            // recorded distinct-node list (confirmed in practice: e.g. work=1
                            // while values={34}, size 1). Rather than crash
                            // (ArrayIndexOutOfBoundsException), treat that as "nothing left to
                            // fix for this job" and move on -- this just skips re-imposing one
                            // specific alternative-node exclusion for it this round, which the
                            // search can still recover through other neighbors/decisions.
                            boolean exhausted = (values == null || work >= values.size());
                            if (exhausted) {
                                if (values == null || work > values.size()) {
                                    System.out.println("### getNeighbor2: skipping data index " + ii + " (i=" + i
                                            + ", data=" + data + ", work=" + work + ", values="
                                            + (values == null ? "null" : values.toString())
                                            + ") -- stale LNS state after a Choco backtrack.");
                                }
                                move = true;
                            } else {
                                for (int k = 0; k < works[ii].length; k++) {
                                    if (jn[ii][k] == values.get(work)) {
                                        jobNodes[ii][k].removeValue(jn[ii][k], this);
                                    } else {
                                        jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                        jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                                    }
                                }
                                work++;
                                if (work == values.size()) {
                                    move = true;
                                }
                            }
                        } else {
                            for (int k = 0; k < works[ii].length; k++) {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                    }
                    if (move) {
                        data = (data + 1) % nb_data;
                        work = 0;
                    }
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                }
            };
        }

        private static INeighbor getNeighbor3(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int node = 0;
                final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    mapping.clear();
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            TIntArrayList list = mapping.get(jn[i][k]);
                            if (list == null) {
                                list = new TIntArrayList();
                                mapping.put(jn[i][k], list);
                            }
                            if (!list.contains(i)) {
                                list.add(i);
                            }
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    node = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    TIntArrayList datas = mapping.get(mapping.keys()[node]);
                    for (int i = 0; i < nb_data; i++) {
                        if (datas.contains(i)) {
                            for (int k = 0; k < works[i].length; k++) {
                                jobNodes[i][k].removeValue(jn[i][k], this);
                            }
                        } else {
                            for (int k = 0; k < works[i].length; k++) {
                                jobNodes[i][k].instantiateTo(jn[i][k], this);
                                //jobStarts[i][k].instantiateTo(sn[i][k], this);
                            }
                        }
                    }
                    node = (node + 1) % mapping.keys().length;
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                }
            };
        }

        private static INeighbor getNeighbor4(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                List<Integer> fixed = IntStream.range(0, nb_data).boxed().collect(Collectors.toList());
                java.util.Random rnd = new java.util.Random(42);
                int nbFixed;
                int round;

                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                    }
                    nbFixed = nb_data - 1;
                    round = 1;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    Collections.shuffle(fixed, rnd);
                    for (int i = 0; i < nbFixed; i++) {
                        for (int k = 0; k < works[i].length; k++) {
                            jobNodes[i][k].instantiateTo(jn[i][k], this);
                            jobStarts[i][k].instantiateTo(sn[i][k], this);
                        }
                    }
                    round++;
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                    if (round % 400 == 0) {
                        nbFixed--;
                    }
                }
            };
        }

        private static INeighbor getNeighbor1bis(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int lim = 0;
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    lim = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    //System.out.printf("1bis : %d\n", lim);
                    for (int i = 0; i < nb_data; i++) {
                        int ii = imaxs[i];
                        for (int k = 0; k < works[ii].length; k++) {
                            if (i != lim) {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                    }
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                    lim = (lim + 1) % nb_data;
                }
            };
        }

        private static INeighbor getNeighbor2bis(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int data = 0;
                int work = 0;
                final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    mapping.clear();
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            TIntArrayList list = mapping.get(i);
                            if (list == null) {
                                list = new TIntArrayList();
                                mapping.put(i, list);
                            }
                            if (!list.contains(jn[i][k])) {
                                list.add(jn[i][k]);
                            }
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    data = 0;
                    work = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    //System.out.printf("2bis : %d - %d\n", imaxs[data], work);
                    boolean move = false;
                    for (int i = 0; i < nb_data; i++) {
                        int ii = imaxs[i];
                        TIntArrayList values = mapping.get(ii);
                        if (i == data) {
                            // Stale cursor guard (same failure getNeighbor2 already guards against):
                            // if a ContradictionException was thrown by a later data's instantiateTo
                            // in a previous call, the trailing `if (move)` never ran, leaving
                            // work == values.size() -- values.get(work) then throws
                            // ArrayIndexOutOfBoundsException. Treat it as "nothing left to fix for
                            // this data" and move on. Never fires on a non-stale cursor.
                            if (values == null || work >= values.size()) {
                                move = true;
                            } else {
                                for (int k = 0; k < works[ii].length; k++) {
                                    if (jn[ii][k] != values.get(work)) {
                                        jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                        //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                                    }
                                }
                                work++;
                                if (work == values.size()) {
                                    move = true;
                                }
                            }
                        } else {
                            for (int k = 0; k < works[ii].length; k++) {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                    }
                    if (move) {
                        data = (data + 1) % nb_data;
                        work = 0;
                    }
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                }
            };
        }

        private static INeighbor getNeighbor3bis(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int node = 0;
                final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    mapping.clear();
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            TIntArrayList list = mapping.get(jn[i][k]);
                            if (list == null) {
                                list = new TIntArrayList();
                                mapping.put(jn[i][k], list);
                            }
                            if (!list.contains(i)) {
                                list.add(i);
                            }
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    node = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    //System.out.printf("3bis : %d\n", mapping.keys()[node]);
                    TIntArrayList datas = mapping.get(mapping.keys()[node]);
                    for (int i = 0; i < nb_data; i++) {
                        if (!datas.contains(i)) {
                            for (int k = 0; k < works[i].length; k++) {
                                jobNodes[i][k].instantiateTo(jn[i][k], this);
                                //jobStarts[i][k].instantiateTo(sn[i][k], this);
                            }
                        }
                    }
                    node = (node + 1) % mapping.keys().length;
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                }
            };
        }

        private static INeighbor getNeighbor4bis(int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
            return new INeighbor() {
                int[][] jn;
                int[][] sn;
                int[] maxs;
                int[] imaxs;
                int node1 = 0;
                int node2 = 0;
                final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
                final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


                @Override
                public void recordSolution() {
                    jn = new int[nb_data][];
                    sn = new int[nb_data][];
                    maxs = new int[nb_data];
                    imaxs = new int[nb_data];
                    mapping.clear();
                    for (int i = 0; i < nb_data; i++) {
                        jn[i] = new int[works[i].length];
                        sn[i] = new int[works[i].length];
                        for (int k = 0; k < works[i].length; k++) {
                            jn[i][k] = jobNodes[i][k].getValue();
                            TIntArrayList list = mapping.get(jn[i][k]);
                            if (list == null) {
                                list = new TIntArrayList();
                                mapping.put(jn[i][k], list);
                            }
                            if (!list.contains(i)) {
                                list.add(i);
                            }
                            sn[i][k] = jobStarts[i][k].getValue();
                        }
                        final int ii = i;
                        maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - 0).max().getAsInt();
                        imaxs[i] = i;
                    }
                    sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                    node1 = 0;
                    node2 = 0;
                }

                @Override
                public void fixSomeVariables() throws ContradictionException {
                    //System.out.printf("3bis : %d\n", mapping.keys()[node]);
                    TIntArrayList datas1 = mapping.get(mapping.keys()[node1]);
                    TIntArrayList datas2 = mapping.get(mapping.keys()[node2]);
                    for (int i = 0; i < nb_data; i++) {
                        if (!datas1.contains(i) && !datas2.contains(i)) {
                            for (int k = 0; k < works[i].length; k++) {
                                jobNodes[i][k].instantiateTo(jn[i][k], this);
                                //jobStarts[i][k].instantiateTo(sn[i][k], this);
                            }
                        }
                    }
                    node2 = (node2 + 1) % mapping.keys().length;
                    if (node2 == 0) {
                        node1 = (node1 + 1) % mapping.keys().length;
                    }
                }

                @Override
                public void loadFromSolution(Solution solution) {

                }

                @Override
                public void restrictLess() {
                }
            };
        }

        public static double transferTime(int job_id, int node_id, int dataSize, int bandwidth, int[][] replicas_location) {
            for(int val:replicas_location[job_id]) {
                if(val == node_id) {
                    // Not a real transfer (data's already there), but NOT free either: a
                    // zero-duration "reuse" task doesn't actually occupy any time in the
                    // node's capacity=1 transfer cumulative, so two different jobs both
                    // already resident there could get instantiated to "reuse" at the exact
                    // same instant with nothing forcing their (chained) work starts apart --
                    // the only thing standing between two different jobs' compute on the same
                    // node, since there's no separate capacity=1 constraint on work itself.
                    // A 1-unit minimum keeps that chain meaningfully sequential; negligible
                    // next to real task durations (tens to hundreds of units).
                    return (double) 1;
                }
            }
            return (double) dataSize / (double) bandwidth;
        }

    }

    static class WorkEntry {
        public Task task;
        public int dataIndex;
        public int nodeIndex;
        public int workIndex;
        public IntVar height;
        public WorkEntry(Task task, int dataIndex, int nodeIndex, int workIndex, IntVar height) {
            this.task = task;
            this.dataIndex = dataIndex;
            this.nodeIndex = nodeIndex;
            this.workIndex = workIndex;
            this.height = height;
        }
    }

    public static class Config {
        public int totalNbComputeNodes;
        public boolean sameStartingTime;
        public double lambdaRate;
        public String jobsFilePath;
    }

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
        public double energyConsumption;
        public double freeTime;
        // Practically unlimited by default: the Python side does not export a real
        // per-node storage capacity yet, so the storage constraint stays non-binding
        // until nodes.json actually provides "storage_capacity".
        public int storageCapacity = Integer.MAX_VALUE / 2;
        public NodeConfig() {}
        public NodeConfig(int bw, double cpu, double energy, double freeTime) {
            this.bandwidth = bw;
            this.computationNodes = cpu;
            this.energyConsumption = energy;
            this.freeTime = freeTime;
        }
    }

    public static void writeWorkConfigCSV(List<SchedulingWithDiffN.WorkConfig> works, String path) throws Exception {
        FileWriter writer = new FileWriter(path);

        // Header
        writer.write("task_index,job_index,start_time,end_time,node_index\n");
        //System.out.println("Writing work configuration ");
        // Rows
        for (SchedulingWithDiffN.WorkConfig w : works) {
            writer.write(
                    w.taskindex + "," +
                            w.jobIndex + "," +
                            w.startTime + "," +
                            w.endTime + "," +
                            w.nodeIndex + "\n"
            );
            //System.out.println("Written work: task " + w.taskindex + " of job " + w.jobIndex + " on node " + w.nodeIndex + " from " + w.startTime + " to " + w.endTime);
        }

        writer.close();
    }

    public static List<String> readLines(String path) throws Exception {
        List<String> lines = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            lines.add(line);
        }
        br.close();
        return lines;
    }

    public static String toJsonArray(List<?> list) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");

        for (int i = 0; i < list.size(); i++) {
            Object val = list.get(i);

            if (val instanceof Number) {
                sb.append("  ").append(val);
            } else {
                sb.append("  \"").append(val.toString()).append("\"");
            }

            if (i < list.size() - 1) sb.append(",");
            sb.append("\n");
        }

        sb.append("]");
        return sb.toString();
    }

    public static void writeTextFile(String path, String content) throws Exception {
        java.nio.file.Files.write(
                java.nio.file.Paths.get(path),
                content.getBytes()
        );
    }

    public static String readFile(String path) throws Exception {
        return new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(path)));
    }

    public static void main(String[] args) throws Exception {

        // ---- Load jobs JSON manually ----
        String jobsText = readFile(MODEL_INPUTS_DIR + "/jobs.json");

        JSONArray jobsArray = new JSONArray(jobsText);
        List<Job> jobs = new ArrayList<>();

        // ---- Load nodes JSON manually --
        String nodesText = readFile(MODEL_INPUTS_DIR + "/nodes.json");
        
        JSONArray nodesArray = new JSONArray(nodesText);
        List<NodeConfig> nodes = new ArrayList<>();

        if (nodesArray.length() == 0) {
            System.err.println("Error: missing informations");
            throw new Exception("Error: missing informations");
        }

        String text = readFile(MODEL_INPUTS_DIR + "/replicas_locations.json");
        
        JSONArray json = new JSONArray(text);

        int[][] replicas_location = new int[json.length()][];
        for (int i = 0; i < json.length(); i++) {
            JSONArray row = json.getJSONArray(i);
            replicas_location[i] = new int[row.length()];
            for (int j = 0; j < row.length(); j++) {
                replicas_location[i][j] = row.getInt(j);
                //System.out.println("Loaded replica location for job " + i + ", node " + j + ": " + row.getInt(j));
            }
            //System.out.println("Loaded replicas location for job " + i + ": " + Arrays.toString(replicas_location[i]));
        }
        
        

        for (int i = 0; i < nodesArray.length(); i++) {
            JSONObject j = nodesArray.getJSONObject(i);

            NodeConfig node = new NodeConfig();
            node.computationNodes = j.getFloat("compute_capacity");
            node.bandwidth = j.getInt("bandwidth");
            node.freeTime = j.getInt("free_time");
            if (j.has("storage_capacity")) {
                node.storageCapacity = j.getInt("storage_capacity");
            }
            // Absent on older nodes.json exports (only this multi-objective file needs it, to
            // price each candidate transfer's energy the same way Tracker.log_transfer /
            // compute_transfer_energy already do on the Python side): default 0 so a stale input
            // just makes energy free rather than failing the whole solve.
            node.energyConsumption = j.has("energy_consumption") ? j.getDouble("energy_consumption") : 0.0;
            nodes.add(node);
            //System.out.println("Loaded node " + i + ": " + node.computationNodes + " CPUs, " + node.bandwidth + " bandwidth, " + node.freeTime + " free time");
        }


        int[] bandwidths = new int[nodes.size()];
        double[] cpus = new double[nodes.size()];
        double[] nodes_free_time = new double[nodes.size()];
        int[] storage_capacity = new int[nodes.size()];
        double[] node_energy_consumption = new double[nodes.size()];
        double[] jobs_arrival_time = new double[jobsArray.length()];

        for (int i = 0; i < nodes.size(); i++) {
            NodeConfig node = nodes.get(i);
            bandwidths[i] = node.bandwidth;
            cpus[i] = node.computationNodes;
            nodes_free_time[i] = node.freeTime;
            storage_capacity[i] = node.storageCapacity;
            node_energy_consumption[i] = node.energyConsumption;
        }

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
            jobs_arrival_time[i] = job.job_arriving_time;
        }
        
                
        int nbData = jobs.size();
        int[] data_sizes = new int[nbData];
        int[][] works = new int[nbData][];

        int ii = 0;
        for (Job job : jobs) {
            data_sizes[ii] = job.datasetSize;

            List<Integer> tasks = new ArrayList<>();
            for (int i = 0; i < job.nbTasks; i++) {
                tasks.add(job.taskDuration);
            }
            works[ii] = tasks.stream().mapToInt(Integer::intValue).toArray();
            //System.out.println("Loaded job " + job.job_id + " with " + job.nbTasks + " tasks.");
            ii++;
            
        }        

        // Call scheduler
        SchedulingWithDiffN.SchedulingResult result = SchedulingWithDiffN.runScheduler(
            jobs,nodes.size(), nbData, data_sizes, works, bandwidths, cpus, storage_capacity, nodes_free_time, replicas_location,null, 0, true,jobs_arrival_time, node_energy_consumption);

        String basePath = MODEL_OUTPUTS_DIR + "/";

        // ---- Pure Java JSON saving ----
        String worksJson = toJsonArray(result.worksExec);
        String transfersJson = toJsonArray(result.transfers);

        //writeTextFile(basePath + "/works_exec_solution.json", worksJson);
        //writeTextFile(basePath + "/transfers_solution.json", transfersJson);

        SchedulingWithDiffN.writeNodeConfigCSV(nodes, basePath + "/nodes_.csv");
        SchedulingWithDiffN.writeTransferConfigCSV(result.transfers, basePath + "/transfers.csv");
        writeWorkConfigCSV(result.worksExec, basePath + "/works.csv");
        SchedulingWithDiffN.writeDeletionConfigCSV(result.deletions, basePath + "/deletions.csv");

    }
}
