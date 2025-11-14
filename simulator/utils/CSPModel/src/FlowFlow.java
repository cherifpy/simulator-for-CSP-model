package simulator.utils.CSPModel.src;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class FlowFlow {

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

    public static void main(String[] args) {


        int seed = 0;
        Random rnd = new Random(seed);

        final int CPU_UNIT = 1;

        // DATA
        int nb_data = 1;
        int[] data_sizes = new int[nb_data];
        for (int i = 0; i < nb_data; i++) {
            data_sizes[i] = 37315; //rnd.nextInt(40960 - 2048 + 1) + 2048; // similar to rnd.choice([2048,40960])
        }
        int nb_min_works = 1;
        int nb_max_works = 20;
        List<List<Integer>> works = new ArrayList<>();
        for (int i = 0; i < nb_data; i++) {
            int nbWorksForData = 4; //rnd.nextInt(nb_max_works - nb_min_works + 1) + nb_min_works;
            List<Integer> wlist = new ArrayList<>();
            
            int works_duration  = 58; //rnd.nextInt(100 - 10 + 1) + 10;
            for (int k = 0; k < nbWorksForData; k++) {
                wlist.add(works_duration);
            }
            works.add(wlist);
        }
        
        
        rnd = new Random(seed);
        // NODES
        int nb_nodes = 20;
        int[] bandwidths = new int[nb_nodes];
        int[] cpus = new int[nb_nodes];
        for (int j = 0; j < nb_nodes; j++) {
            bandwidths[j] = rnd.nextInt(800 - 12 + 1) + 12; // [12,800]
            cpus[j] = (rnd.nextInt(10) + 1) * CPU_UNIT;     // [1, 10]
        }

        // compute an upper bound on makespan (same idea as python)
        long makespanLong = 0;
        
        long sumData = 0;
        for (int s : data_sizes) sumData += s;
        
        int minBandwidth = Integer.MAX_VALUE;
        for (int b : bandwidths) if (b < minBandwidth) minBandwidth = b;
        System.err.println("minBandwidth="+minBandwidth);
        makespanLong = sumData / Math.max(1, minBandwidth);
        
        long totalWork = 0;
        for (List<Integer> wl : works) for (int w : wl) totalWork += w;
        
        int maxCpu = 0;
        for (int c : cpus) if (c > maxCpu) maxCpu = c;
        
        makespanLong += totalWork * CPU_UNIT * Math.max(1, maxCpu);
        makespanLong *= 2;

        
        int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE);
        
        System.out.println("Computed makespan upper bound: " + makespan);
        // print inputs (summary)
        System.out.println("DATA");
        for (int i = 0; i < nb_data; i++) {
            System.out.println(" Data " + i + ": size=" + data_sizes[i] + " MB, works=" + works.get(i));
        }
        System.out.println("NODES");
        for (int j = 0; j < nb_nodes; j++) {
            System.out.println(" Node " + j + ": bandwidth=" + bandwidths[j] + " MB/s, cpu=" + cpus[j] + " units/s");
        }

        // ----- MODEL -----
        Model model = new Model("Bag of Tasks Scheduling (Java)");

        // Arrays for transfer tasks and heights
        Task[][] transferTasks = new Task[nb_nodes][nb_data];
        IntVar[][] transferHeights = new IntVar[nb_nodes][nb_data];

        // For storing work tasks
        List<WorkEntry> workTasks = new ArrayList<>();

        // Create transfer tasks: one per (node, data)
        for (int j = 0; j < nb_nodes; j++) {
            for (int i = 0; i < nb_data; i++) {
                
                IntVar s = model.intVar("start_transfer_d" + i + "_n" + j, 0, makespan);
                int d = (int) Math.ceil((double) data_sizes[i] / (double) bandwidths[j]); 
                IntVar durationVar = model.intVar(d);
                IntVar end = model.intVar("end_transfer_d" + i + "_n" + j, 0, makespan);
                Task t = new Task(s, durationVar, end);
                IntVar h = model.intVar("height_transfer_d" + i + "_n" + j, 0, 1);
                transferTasks[j][i] = t;
                transferHeights[j][i] = h;
            }
        }

        // Create work tasks: for each node, each data, each wor 
        for (int j = 0; j < nb_nodes; j++) {
            for (int i = 0; i < nb_data; i++) {
                List<Integer> wl = works.get(i);
                for (int k = 0; k < wl.size(); k++) {
                    int w = wl.get(k);
                    IntVar s = model.intVar("start_work_d" + i + "_w" + k + "_n" + j, 0, makespan);
                    int d =  w * cpus[j];
                    IntVar durationVar = model.intVar(d);
                    IntVar end = model.intVar("end_work_d" + i + "_w" + k + "_n" + j, 0, makespan);
                    Task t = new Task(s, durationVar, end);
                    IntVar h = model.intVar("height_work_d" + i + "_w" + k + "_n" + j, 0, 1);
                    workTasks.add(new WorkEntry(t, i, j, k, h));
                }
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
            System.out.printf("Task_%d -- duration= %s%n", j, tasksForNode[0].getDuration());
            model.cumulative(tasksForNode, heightsForNode, model.intVar(1)).post();
        }

        // At least one transfer per data (sum over nodes heights[j][i] >= 1)
        for (int i = 0; i < nb_data; i++) {
            IntVar[] arr = new IntVar[nb_nodes];
            for (int j = 0; j < nb_nodes; j++) arr[j] = transferHeights[j][i];
            model.sum(arr, ">=", 1).post();
        }

        // A work can start only after the corresponding transfer is finished on that node,
        // and only if the transfer happened (height)
        for (WorkEntry we : workTasks) {
            Task wt = we.task;
            int i = we.dataIndex;
            int j = we.nodeIndex;
            IntVar h = we.height;
            // wt.getStart() and transferTasks[j][i].getEnd() are IntVar
            model.arithm(wt.getStart(), ">=", transferTasks[j][i].getEnd()).post();
            model.arithm(h, "<=", transferHeights[j][i]).post();
        }
        // if a transfer happens, then at least one work must happen on that node for that data
        for (int j = 0; j < nb_nodes; j++) {
            for (int i = 0; i < nb_data; i++) {
                List<IntVar> workHeightsForDataNode = new ArrayList<>();
                for (WorkEntry we : workTasks) {
                    if (we.dataIndex == i && we.nodeIndex == j) {
                        workHeightsForDataNode.add(we.height);
                    }
                }
                if (!workHeightsForDataNode.isEmpty()) {
                    IntVar[] whArr = workHeightsForDataNode.toArray(new IntVar[0]);
                    model.max(transferHeights[j][i], whArr).post();
                }
            }
        }

        // Cumulative constraint on nodes for works (capacity = 1)
        for (int j = 0; j < nb_nodes; j++) {
            // collect tasks and heights for works whose nodeIndex == j
            List<Task> tlist = new ArrayList<>();
            List<IntVar> hlist = new ArrayList<>();
            for (WorkEntry we : workTasks) {
                if (we.nodeIndex == j) {
                    tlist.add(we.task);
                    hlist.add(we.height);
                }
            }
            if (!tlist.isEmpty()) {
                Task[] tArr = tlist.toArray(new Task[0]);
                IntVar[] hArr = hlist.toArray(new IntVar[0]);
                model.cumulative(tArr, hArr, model.intVar(1)).post();
            }
        }

        // Each work must be done exactly once (sum of heights for a given (data i, work k) across nodes == 1)
        for (int i = 0; i < nb_data; i++) {
            int nbWorksForData = works.get(i).size();
            for (int k = 0; k < nbWorksForData; k++) {
                List<IntVar> hs = new ArrayList<>();
                for (WorkEntry we : workTasks) {
                    if (we.dataIndex == i && we.workIndex == k) hs.add(we.height);
                }
                if (!hs.isEmpty()) {
                    IntVar[] hsArr = hs.toArray(new IntVar[0]);
                    model.sum(hsArr, "=", 1).post();
                }
            }
        }

        // ----- OBJECTIVE -----
        // makespan var and ensure it's >= all ends
        IntVar makespanVar = model.intVar("makespan", 0, makespan);
        // collect all work ends
        IntVar[] endsArr = new IntVar[workTasks.size()];
        int index = 0;
        for (WorkEntry we : workTasks) {
            endsArr[index] = model.intVar(0, we.task.getEnd().getUB());
            model.times(we.task.getEnd(), we.height, endsArr[index]).post();
            index++;
        }
        // model.max(makespanVar, endsArr).post(); -- post max constraint between makespanVar and all ends
        model.max(makespanVar, endsArr).post();
        // minimize makespan
        model.setObjective(false, makespanVar); // false => MINIMIZE (see Choco API)

        // SOLVER & strategy: branch first on binary placement vars (heights), then on makespan
        

        Solver solver = model.getSolver();

        // 1) Build prioritized decision var list: all transfer heights, all work heights, then makespan
        //List<IntVar> dv = new ArrayList<>();
        //for (int j = 0; j < nb_nodes; j++) for (int i = 0; i < nb_data; i++) dv.add(transferHeights[j][i]);
        //for (WorkEntry we : workTasks) dv.add(we.height);
        //dv.add(makespanVar);
        //IntVar[] searchVars = dv.toArray(new IntVar[0]);

        // 2) Set a concrete search strategy (branch on binary placement vars first).
        //    minDomLBSearch is a good general-purpose heuristic (min domain, choose LB).
        //solver.setSearch(minDomLBSearch(endsArr)); 

      
        //solver.clearLimits(); // remove any previous limit
        //solver.limitTime("300s");

        // 4) Call the optimization routine that searches for an optimum.
        //findOptimalSolution(objective, Model.MINIMIZE)
        
        
        // ----- SOLVER -----
        //Solver solver = model.getSolver();
        solver.showShortStatistics();

        boolean found = false;
        while (solver.solve()) {
            found = true;
            System.out.println("Solution found with makespan = " + makespanVar.getValue());
            
            for (int j = 0; j < nb_nodes; j++) {
                System.out.println("Node " + j + ":");
                for (int i = 0; i < nb_data; i++) {
                    if (transferHeights[j][i].getValue() == 1) {
                        System.out.println("  Data " + i + " transferred from "
                                + transferTasks[j][i].getStart() + " to "
                                + transferTasks[j][i].getEnd());
                    }
                }
                
                for (WorkEntry we : workTasks) {
                    if (we.nodeIndex == j && we.height.getValue() == 1) {
                        System.out.println("  Work " + we.workIndex + " of Data " + we.dataIndex
                                + " processed from " + we.task.getStart() + " to " + we.task.getEnd());
                    }
                }
            }
            System.out.println();
        }
        if (!found) {
            System.out.println("No solution found");
        }

    }
}
