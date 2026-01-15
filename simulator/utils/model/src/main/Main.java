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

public class Main {


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

        public static class SchedulingResult {
            public List<TransferConfig> transfers = new ArrayList<>();
            public List<WorkConfig> worksExec = new ArrayList<>();

            public SchedulingResult(List<TransferConfig> transfers, List<WorkConfig> worksExec) {
                this.transfers = transfers;
                this.worksExec = worksExec;
            }
        }

        public static void writeNodeConfigCSV(List<Main.NodeConfig> nodes, String path) throws Exception {
            FileWriter writer = new FileWriter(path);

            // header
            writer.write("bandwidth,computation_nodes,energy_consumption\n");

            // rows
            for (Main.NodeConfig n : nodes) {
                writer.write(
                        n.bandwidth + "," +
                                n.computationNodes + "," +
                                n.energyConsumption + "\n"
                );
            }

            writer.close();
        }

        public static void writeTransferConfigCSV(List<SchedulingWithDiffN.TransferConfig> transfers, String path) throws Exception {
            FileWriter writer = new FileWriter(path);

            // header
            writer.write("job_index,start_time,end_time,node_index\n");

            // rows
            for (TransferConfig t : transfers) {
                writer.write(
                        t.jobIndex + "," +
                                t.startTime + "," +
                                t.endTime + "," +
                                t.nodeIndex + "\n"
                );
            }

            writer.close();
        }

        public static SchedulingResult runScheduler(List<Main.Job>  jobs,int nb_nodes, int nb_data, int[] data_sizes, int[][] works, int[] bandwidths, double[] cpus, double[] starting_times, int[][]  replicas_location, Model[] models, int pos, boolean solve) {
            final int CPU_UNIT = 1; // to scale cpu speeds
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

            makespanLong += 0;
            makespanLong += totalWork * CPU_UNIT * Math.max(1, maxCpu);
            makespanLong *= 2;


            int makespan = 6_000;//(int) Math.min(makespanLong, Integer.MAX_VALUE);

            //System.out.println("Computed makespan upper bound: " + makespan);
            // print inputs (summary)
            //System.out.println("DATA");
            /*for (int i = 0; i < nb_data; i++) {
                System.out.println(" Data " + i + ": size=" + data_sizes[i] + " MB, works=" + Arrays.toString(works[i]) + " arrival=" + starting_times[i]);
            }
            System.out.println("NODES");
            for (int j = 0; j < nb_nodes; j++) {
                System.out.println(" Node " + j + ": bandwidth=" + bandwidths[j] + " MB/s, cpu=" + cpus[j] + " units/s");
            }*/

            // ----- MODEL -----
            Model model = new Model("Bag of Tasks Scheduling (Java)",
            Settings.dev()
                    .setLCG(false)
                    .setWarnUser(true));

            // Arrays for transfer tasks and heights
            Task[][] transferTasks = new Task[nb_nodes][nb_data];
            BoolVar[][] transferHeights = new BoolVar[nb_nodes][nb_data];

            // Create transfer tasks: one per (node, data)
            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {

                    IntVar s = model.intVar("start_transfer_d" + i + "_n" + j, (int) starting_times[j], makespan,true);
                    
                    int d = (int) Math.ceil(transferTime(i, j, data_sizes[i], bandwidths[j], replicas_location));
                    
                    IntVar durationVar = model.intVar(d);
                    IntVar end = model.intVar("end_transfer_d" + i + "_n" + j, (int) starting_times[j] + d, makespan,true);
                    BoolVar h = model.boolVar("height_transfer_d" + i + "_n" + j);
                    Task t = new Task(s, durationVar, end);
                    transferTasks[j][i] = t;
                    transferHeights[j][i] = h;
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
                System.out.println("Creating work tasks for data " + i + " with " + wl.length + " works.");
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
                    jobNodes[i][k] = model.intVar("node_work_d" + i + "_w" + k, 0, nb_nodes - 1);
                    model.element(jobDurations[i][k], durations, jobNodes[i][k]).post();
                    jobEnds[i][k] = model.intVar("end_work_d" + i + "_w" + k, 0, makespan,true);//(int) starting_times[j]
                    model.arithm(jobStarts[i][k], "+", jobDurations[i][k], "=", jobEnds[i][k]).post();
                    for (int j = 0; j < nb_nodes; j++) {
                        BoolVar jOnN = jobNodes[i][k].eq(j).boolVar();
                        // A work can start only after the corresponding transfer is finished on that node,
                        model.impXrelYC(jobStarts[i][k], ">=", transferTasks[j][i].getEnd(), 0, jOnN);
                    }
                }
            }

            //  if a transfer happens, then at least one work must happen on that node for that data
            for (int i = 0; i < nb_data; i++) {
                int[] wl = works[i];
                IntVar[] counters = new IntVar[nb_nodes];
                for (int j = 0; j < nb_nodes; j++) {
                    counters[j] = model.intVar(0, wl.length);
                    model.count(j, jobNodes[i], counters[j]).post();
                    model.reifXrelC(counters[j], ">=", 1, transferHeights[j][i]);
                }
                model.sum(counters, "=", wl.length).post();
                for (int k = 0; k < wl.length - 1; k++) {
                    jobNodes[i][k].eq(jobNodes[i][k + 1]).imp(jobStarts[i][k].lt(jobStarts[i][k + 1])).post();
                    // very strict :
                    jobNodes[i][k].le(jobNodes[i][k + 1]).post();
                }
            }


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


            // Transfer_time <= factor * sum(execution_time)
            int factor=1;
            for (int i = 0; i < nb_data && factor > 0; i++) {
                for (int j = 0; j < nb_nodes; j++) {
                    IntVar[] executions = new IntVar[works[i].length];
                    for (int k = 0; k < executions.length; k++) {
                        executions[k] = model.isEq(jobNodes[i][k], j).mul(jobDurations[i][k]).intVar();
                    }
                    model.sum(executions, ">=", transferHeights[j][i], transferTasks[j][i].getDuration().getValue() * factor).post();
                }
            }

            // ----- OBJECTIVE Make span-----
            //makespan var and ensure it's >= all end
            boolean makespan_obj = false;
            final IntVar[] objectives = new IntVar[2];
            if (makespan_obj) {
                /*IntVar makespanVar = model.intVar("makespan", 0, makespan);

                // collect all work ends
                IntVar[] endsArr = new IntVar[workTasks.size()];
                int index = 0;
                for (Main.WorkEntry we : workTasks) {
                    endsArr[index] = model.intVar(0, we.task.getEnd().getUB());
                    model.impXrelYC(we.task.getEnd(), "=", endsArr[index], 0, we.height.asBoolVar());
                    model.impXrelC(endsArr[index], "=", 0, we.height.asBoolVar().not());
                    index++;
                }
                //
                new Constraint("max", new PropMax(endsArr, makespanVar)).post();
                //model.max(makespanVar, endsArr).post();// -- post max constraint between makespanVar and all ends
                // minimize makespan
                model.setObjective(false, makespanVar); // false => MINIMIZE (see Choco API)*/
            } else {

                IntVar[] all_flow_time = new IntVar[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    IntVar end_time = model.intVar(0, makespan);
                    model.max(end_time, jobEnds[i]).post();
                    //IntVar flow = model.intVar(0, makespan);
                    //model.arithm(end_time, "-", flow, "=", starting_times[i]).post();
                    IntVar flow = model.intView(1, end_time, 0 ); //-starting_times[i]
                    all_flow_time[i] = flow;
                }
                IntVar maxFlowTime = model.intVar("max_flow_time", 0, 999_999);
                model.max(maxFlowTime, all_flow_time).post();
                IntVar sumFlowTime = model.intVar("sum_flow_time", 0, 999_999);
                model.sum(all_flow_time, "=", sumFlowTime).post();
                //model.setObjective(false, maxFlowTime);
                objectives[1] = maxFlowTime;
                objectives[0] = sumFlowTime;
            }

            //----- SOLVER -----
            Solver solver = model.getSolver();
            List<TransferConfig> transfersList = new ArrayList<>();
            List<WorkConfig> worksList = new ArrayList<>();
            model.displayVariableOccurrences();
            model.displayPropagatorOccurrences();

            IntVar[] decisionVars = decisionVariables(nb_nodes, nb_data, works, jobNodes, jobStarts, transferHeights, transferTasks);
            hints(nb_nodes, nb_data, data_sizes, works, cpus, solver, jobNodes);

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

            solver.limitTime("300s");
            
            boolean[] found = {false};
            solver.onSolution(() -> {
                        
                    found[0] = true;

                    transfersList.clear();
                    worksList.clear();

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
                            }
                        }
                    }

                /*System.out.printf("%d;%d;%.2f;%d\n",
                        objectives[0].getValue(), objectives[1].getValue(), solver.getTimeCount(), solver.getSolutionCount());*/
            });

            solver.findOptimalSolution(objectives[0], false);
            if (!found[0]) {
                System.out.println("No solution found");
            }

            SchedulingResult result = new SchedulingResult(transfersList, worksList);

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
                int ii = cidx[i];
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
                    return (double) 0;
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
    }

    public static class NodeConfig {
        public int bandwidth;
        public double computationNodes;
        public double energyConsumption;
        public double freeTime;
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

        // Rows
        for (SchedulingWithDiffN.WorkConfig w : works) {
            writer.write(
                    w.taskindex + "," +
                            w.jobIndex + "," +
                            w.startTime + "," +
                            w.endTime + "," +
                            w.nodeIndex + "\n"
            );
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
        String jobsText = readFile("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/jobs.json");

        JSONArray jobsArray = new JSONArray(jobsText);
        List<Job> jobs = new ArrayList<>();

        // ---- Load nodes JSON manually --
        String nodesText = readFile("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/nodes.json");
        
        JSONArray nodesArray = new JSONArray(nodesText);
        List<NodeConfig> nodes = new ArrayList<>();

        if (nodesArray.length() == 0) {
            System.err.println("Error: missing informations");
            throw new Exception("Error: missing informations");
        }

        String text = readFile("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/replicas_locations.json");
        JSONArray json = new JSONArray(text);

        int[][] replicas_location = new int[json.length()][];
        for (int i = 0; i < json.length(); i++) {
            JSONArray row = json.getJSONArray(i);
            replicas_location[i] = new int[row.length()];
            for (int j = 0; j < row.length(); j++) {
                replicas_location[i][j] = row.getInt(j);
            }
        }
        
        

        for (int i = 0; i < nodesArray.length(); i++) {
            JSONObject j = nodesArray.getJSONObject(i);

            NodeConfig node = new NodeConfig();
            node.computationNodes = j.getInt("compute_capacity");
            node.bandwidth = j.getInt("bandwidth");
            node.freeTime = j.getInt("free_time");
            nodes.add(node);
        }


        int[] bandwidths = new int[nodes.size()];
        double[] cpus = new double[nodes.size()];
        double[] nodes_free_time = new double[nodes.size()];

        
        for (int i = 0; i < nodes.size(); i++) {
            NodeConfig node = nodes.get(i);
            bandwidths[i] = node.bandwidth;
            cpus[i] = node.computationNodes;
            nodes_free_time[i] = node.freeTime;
        }

        for (int i = 0; i < jobsArray.length(); i++) {
            JSONObject j = jobsArray.getJSONObject(i);

            Job job = new Job();
            job.datasetSize = j.getInt("dataset_size");
            job.nbTasks = j.getInt("nb_tasks");
            job.taskDuration = j.getInt("task_duration");
            job.job_id = j.getInt("job_id");
            jobs.add(job);
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
            System.err.println("Loaded job " + job.job_id + " with " + job.nbTasks + " tasks.");
            ii++;
            
        }        

        // Call scheduler
        SchedulingWithDiffN.SchedulingResult result = SchedulingWithDiffN.runScheduler(
            jobs,nodes.size(), nbData, data_sizes, works, bandwidths, cpus, nodes_free_time, replicas_location,null, 0, true);

        String basePath = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/outputs/";

        // ---- Pure Java JSON saving ----
        String worksJson = toJsonArray(result.worksExec);
        String transfersJson = toJsonArray(result.transfers);

        //writeTextFile(basePath + "/works_exec_solution.json", worksJson);
        //writeTextFile(basePath + "/transfers_solution.json", transfersJson);

        SchedulingWithDiffN.writeNodeConfigCSV(nodes, basePath + "/nodes_.csv");
        SchedulingWithDiffN.writeTransferConfigCSV(result.transfers, basePath + "/transfers.csv");
        writeWorkConfigCSV(result.worksExec, basePath + "/works.csv");

    }
}
