//package simulator.utils.CSPModel;
package main;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

import static org.chocosolver.solver.search.strategy.Search.*;

import gnu.trove.TIntCollection;
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

public class Main {

    public static class Scheduling {

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
            public List<Scheduling.TransferConfig> transfers = new ArrayList<>();
            public List<Scheduling.WorkConfig> worksExec = new ArrayList<>();

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

        public static void writeTransferConfigCSV(List<Scheduling.TransferConfig> transfers, String path) throws Exception {
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

        public static double transferTime(int job_id, int node_id, int dataSize, int bandwidth, int[][] replicas_location) {
            for(int val:replicas_location[job_id]) {
                if(val == node_id) {
                    return (double) 0;
                }
            }
            return (double) dataSize / (double) bandwidth;
        }

        public static SchedulingResult runScheduler(List<Main.Job>  jobs,int nb_nodes, int nb_data, int[] data_sizes, int[][] works, int[] bandwidths, double[] cpus, double[] starting_times, int[][] replicas_location) {
            
            for(Job job: jobs) {
                System.out.println("Job id: "+ job.job_id + " dataset size: "+ job.datasetSize + " nb tasks: "+ job.nbTasks + " task duration: "+ job.taskDuration);
            }       
            
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


            int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE);

            // ----- MODEL -----
            Model model = new Model("Bag of Tasks Scheduling (Java)");

            // Arrays for transfer tasks and heights
            Task[][] transferTasks = new Task[nb_nodes][nb_data];
            IntVar[][] transferHeights = new IntVar[nb_nodes][nb_data];

            

            // For storing work tasks
            List<Main.WorkEntry> workTasks = new ArrayList<>();

            // Create transfer tasks: one per (node, data)
            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {

                    IntVar s = model.intVar("start_transfer_d" + i + "_n" + j, (int) starting_times[j], makespan);
                    //int d = (int) Math.ceil((double) data_sizes[i] / (double) bandwidths[j]);
                    //transferTime(int job_id, int node_id, int dataSize, int bandwidth, int[][] replicas_location)
                    int d = (int) Math.ceil(transferTime(i, j, data_sizes[i], bandwidths[j], replicas_location));
                    IntVar durationVar = model.intVar(d);
                    IntVar end = model.intVar("end_transfer_d" + i + "_n" + j, (int) starting_times[j], makespan);
                    Task t = new Task(s, durationVar, end);
                    IntVar h = model.intVar("height_transfer_d" + i + "_n" + j, 0, 1);
                    transferTasks[j][i] = t;
                    transferHeights[j][i] = h;
                }
            }

            // Create work tasks: for each node, each data, each wor
            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {
                    int[] wl = works[i];
                    for (int k = 0; k < wl.length; k++) {
                        int w = wl[k];
                        IntVar s = model.intVar("start_work_d" + i + "_w" + k + "_n" + j, (int) starting_times[j], makespan);
                        int d = (int) ((int) w * cpus[j]);
                        IntVar durationVar = model.intVar(d);
                        IntVar end = model.intVar("end_work_d" + i + "_w" + k + "_n" + j, (int) starting_times[j], makespan);
                        Task t = new Task(s, durationVar, end);
                        IntVar h = model.intVar("height_work_d" + i + "_w" + k + "_n" + j, 0, 1);
                        workTasks.add(new Main.WorkEntry(t, i, j, k, h));
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
                // System.out.printf("Task_%d -- duration= %s%n", j, tasksForNode[0].getDuration());
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
            for (Main.WorkEntry we : workTasks) {
                Task wt = we.task;
                int i = we.dataIndex;
                int j = we.nodeIndex;
                IntVar h = we.height;
                // wt.getStart() and transferTasks[j][i].getEnd() are IntVar
                //IntVar t_start_time = model.intVar(0,makespan);
                //model.times(wt.getStart(), h, t_start_time).post();
                model.arithm(wt.getStart(), ">=", transferTasks[j][i].getEnd()).post();
                model.arithm(h, "<=", transferHeights[j][i]).post();
            }
            // if a transfer happens, then at least one work must happen on that node for that data
            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {
                    List<IntVar> workHeightsForDataNode = new ArrayList<>();
                    for (Main.WorkEntry we : workTasks) {
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
                for (Main.WorkEntry we : workTasks) {
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
                int nbWorksForData = works[i].length;
                for (int k = 0; k < nbWorksForData; k++) {
                    List<IntVar> hs = new ArrayList<>();
                    for (Main.WorkEntry we : workTasks) {
                        if (we.dataIndex == i && we.workIndex == k) hs.add(we.height);
                    }
                    if (!hs.isEmpty()) {
                        IntVar[] hsArr = hs.toArray(new IntVar[0]);
                        model.sum(hsArr, "=", 1).post();
                    }
                }
            }


            // ----- OBJECTIVE Make span-----
            //makespan var and ensure it's >= all end
            boolean makespan_obj = false;
            if (makespan_obj) {
                IntVar makespanVar = model.intVar("makespan", 0, makespan);

                // collect all work ends
                IntVar[] endsArr = new IntVar[workTasks.size()];
                int index = 0;
                for (Main.WorkEntry we : workTasks) {
                    endsArr[index] = model.intVar(0, we.task.getEnd().getUB());
                    model.times(we.task.getEnd(), we.height, endsArr[index]).post();
                    //IntVar start = model.intVar(starting_times[we.dataIndex * (-1)]);
                    index++;
                }
                //
                model.max(makespanVar, endsArr).post();// -- post max constraint between makespanVar and all ends
                // minimize makespan
                model.setObjective(false, makespanVar); // false => MINIMIZE (see Choco API)
            } else {


                IntVar[][] all_end_times = new IntVar[nb_data][];

                for (int i = 0; i < nb_data; i++) {
                    all_end_times[i] = new IntVar[works[i].length * nb_nodes];
                }
                IntVar[] all_flow_time = new IntVar[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    IntVar start_time = model.intVar(0);
                    int k = 0;
                    for (Main.WorkEntry we : workTasks) {
                        if (we.dataIndex == i) {
                            IntVar tmp_end = model.intVar(0, makespan);
                            //IntVar J = model.intVar(1);
                            model.times(we.height, we.task.getEnd(), tmp_end).post();
                            all_end_times[i][k] = tmp_end;
                            k++;
                        }
                    }
                    IntVar end_time = model.intVar(0, makespan);
                    model.max(end_time, all_end_times[i]).post();
                    IntVar flow = model.intVar(0, makespan);
                    IntVar[] to_sum = new IntVar[2];
                    to_sum[0] = start_time;
                    to_sum[1] = end_time;
                    model.sum(to_sum, "=", flow).post();
                    all_flow_time[i] = flow;
                }

                boolean avgFlowTimeBolean = true;
                IntVar maxFlowTime = model.intVar("max_flow_time", 0, makespan);
                model.max(maxFlowTime, all_flow_time).post();


                IntVar totalFlowTime = model.intVar("totalFlowTime", 0, makespan * 2);
                model.sum(all_flow_time, "=", totalFlowTime).post();
                IntVar avgFlowTime = model.intVar("avgFlowTime", 0, makespan);
                IntVar nb_data_var =  model.intVar(nb_data);
                model.div(totalFlowTime, nb_data_var, avgFlowTime).post();


                if (avgFlowTimeBolean) {
                    model.setObjective(false, avgFlowTime);

                }else{
                    model.setObjective(false, maxFlowTime);
                }
            }


            Solver solver = model.getSolver();
            solver.showShortStatistics();
            solver.limitTime("90s");
            boolean found = false;

            List<TransferConfig> transfersList = new ArrayList<>();
            List<WorkConfig> worksList = new ArrayList<>();

            List<IntVar> vars = new ArrayList<>();

            for (int j = 0; j < nb_nodes; j++) {
                for (int i = 0; i < nb_data; i++) {
                    vars.add(transferHeights[j][i]);
                    vars.add(transferTasks[j][i].getStart());
                }
            }
            for (Main.WorkEntry we : workTasks) {
                vars.add(we.height);
                vars.add(we.task.getStart());
            }

            // first branch on transfer heights, then start
            IntVar[] decisionVars = vars.toArray(new IntVar[0]);
            BlackBoxConfigurator bb = BlackBoxConfigurator.init();
            // variable selection
            bb.setIntVarStrategy(vs -> Search.roundRobinSearch(decisionVars))
                    .setRestartPolicy(s -> new Restarter(
                            new InnerOuterCutoff(300, 1.01, 1.01),
                            c -> s.getFailCount() >= c, 50_000, true))
                    .setNogoodOnRestart(true)
                    .setRestartOnSolution(true)
                    .setRefinedPartialAssignmentGeneration(true)
                    .setExcludeObjective(true)
                    .setExcludeViews(false);
            bb.make(model);
            int time = 600;
            System.out.println("Starting solver with time limit: " + time + "s");
            solver.limitTime(time + "s");
            while (solver.solve()) {

                found = true;
                if (makespan_obj) {
                    System.out.println("Solution found with max makespan = " + model.getObjective().asIntVar().getValue());
                } else {
                    System.out.println("Solution found with max Flow Time = " + model.getObjective().asIntVar().getValue());
                }

                transfersList.clear();
                worksList.clear();

                for (int j = 0; j < nb_nodes; j++) {
                    //System.out.println("Node " + j + ":");
                    for (int i = 0; i < nb_data; i++) {
                        if (transferHeights[j][i].getValue() == 1) {
                            //System.out.println("  Data " + i + " transferred from "+ transferTasks[j][i].getStart() + " to "+ transferTasks[j][i].getDuration());

                            TransferConfig tmp_transfer = new TransferConfig(jobs.get(i).job_id, transferTasks[j][i].getStart().getValue(), transferTasks[j][i].getEnd().getValue(), j);
                            transfersList.add(tmp_transfer);
                        }
                    }

                    for (Main.WorkEntry we : workTasks) {
                        if (we.nodeIndex == j && we.height.getValue() == 1) {
                            //System.out.println("  Work " + we.workIndex + " of Data " + we.dataIndex+ " processed from " + we.task.getStart() + " to " + we.task.getDuration());
                            
                            WorkConfig tmp_work = new WorkConfig(we.workIndex, jobs.get(we.dataIndex).job_id, we.task.getStart().getValue(), we.task.getEnd().getValue(), j);
                            worksList.add(tmp_work);
                        }

                    }
                }
                System.out.println();
            }
            if (!found) {
                System.out.println("No solution found");
            }

            SchedulingResult result = new SchedulingResult(transfersList, worksList);

            return result;
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

    public static void writeWorkConfigCSV(List<Scheduling.WorkConfig> works, String path) throws Exception {
        FileWriter writer = new FileWriter(path);

        // Header
        writer.write("task_index,job_index,start_time,end_time,node_index\n");

        // Rows
        for (Scheduling.WorkConfig w : works) {
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
            ii++;
        }        

        Scheduling sch = new Scheduling();
        // Call scheduler
        Scheduling.SchedulingResult result = Scheduling.runScheduler(jobs,nodes.size(), nbData, data_sizes, works, bandwidths, cpus, nodes_free_time, replicas_location);

        String basePath = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/outputs/";

        // ---- Pure Java JSON saving ----
        String worksJson = toJsonArray(result.worksExec);
        String transfersJson = toJsonArray(result.transfers);

        //writeTextFile(basePath + "/works_exec_solution.json", worksJson);
        //writeTextFile(basePath + "/transfers_solution.json", transfersJson);

        Scheduling.writeNodeConfigCSV(nodes, basePath + "/nodes_.csv");
        Scheduling.writeTransferConfigCSV(result.transfers, basePath + "/transfers.csv");
        writeWorkConfigCSV(result.worksExec, basePath + "/works.csv");

    }
}
