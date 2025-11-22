//package simulator.utils.CSPModel;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;

import model.Scheduling;

import static org.chocosolver.solver.search.strategy.Search.*;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class Main {

    
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
        public int datasetSize;
        public int nbTasks;
        public int taskDuration;
    }

    public static class NodeConfig {
        public int bandwidth;
        public double computationNodes;
        public double energyConsumption;

        public NodeConfig(int bw, double cpu, double energy) {
            this.bandwidth = bw;
            this.computationNodes = cpu;
            this.energyConsumption = energy;
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

    public static List<NodeConfig> loadNodeConfigCSV(String path) throws Exception {
        List<String> lines = readLines(path);
        List<NodeConfig> list = new ArrayList<>();

        // Skip header (line 0)
        for (int i = 1; i < lines.size(); i++) {
            String[] parts = lines.get(i).split(",");

            int bw = Integer.parseInt(parts[0]);
            double cpu = Double.parseDouble(parts[1]);
            double energy = Double.parseDouble(parts[2]);

            list.add(new NodeConfig(bw, cpu, energy));
        }

        return list;
    }

    /* ============================================================
        =================   GENERATE INFRA   ========================
        ============================================================ */
    public static List<NodeConfig> generateHeterogeneousInfrastructureEquilibre(Config config) {

        int nbNode = config.totalNbComputeNodes;

        Map<Integer, Double> proportions = Map.of(
                0, 0.30,
                1, 0.30,
                2, 0.20,
                3, 0.20
        );

        Map<Integer, Map<String, double[]>> cats = new HashMap<>();
        cats.put(0, Map.of("VCPU", new double[]{1, 5.1}, "BW", new double[]{12, 128}));
        cats.put(1, Map.of("VCPU", new double[]{6, 10}, "BW", new double[]{12, 128}));
        cats.put(2, Map.of("VCPU", new double[]{1, 5.1}, "BW", new double[]{600, 800}));
        cats.put(3, Map.of("VCPU", new double[]{6, 10}, "BW", new double[]{600, 800}));

        Random random = new Random(42);
        List<NodeConfig> nodes = new ArrayList<>();

        for (int cat = 0; cat < 4; cat++) {
            int count = (int) (proportions.get(cat) * nbNode);

            for (int i = 0; i < count; i++) {
                double vcpu = cats.get(cat).get("VCPU")[0] +
                        random.nextDouble() * (cats.get(cat).get("VCPU")[1] - cats.get(cat).get("VCPU")[0]);

                int bw = (int) (cats.get(cat).get("BW")[0] +
                        random.nextDouble() * (cats.get(cat).get("BW")[1] - cats.get(cat).get("BW")[0]));

                double energy = 0.1 + random.nextDouble() * (2.1 - 0.1);

                nodes.add(new NodeConfig(bw, vcpu, energy));
            }
        }

        return nodes;
    }

    /* ============================================================
       ======================   SCHEDULER   ========================
       ============================================================ */


    /* ============================================================
       ========================= MAIN ==============================
       ============================================================ */

    private static final Logger logger = Logger.getLogger(Main.class.getName());

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
        System.out.println("Hello World!");
        // Fichiers utilisés (vous pouvez modifier comme vous voulez)
        String configPath = "config.json";

        // Configure logging
        Logger root = Logger.getLogger("");
        root.setLevel(Level.INFO);

        Random random = new Random(42);

        // ---- Load config JSON manually ----
        String configText = readFile(configPath);
        JSONObject configJson = new JSONObject(configText);

        Config config = new Config();
        config.totalNbComputeNodes = configJson.getInt("total_nb_compute_nodes");
        config.sameStartingTime = configJson.getBoolean("same_starting_time");
        config.lambdaRate = configJson.getDouble("lambda_rate");

        // ---- Load jobs JSON manually ----
        config.jobsFilePath = "workloads/jobs-a-a-20.json";
        String jobsText = readFile(config.jobsFilePath);

        JSONArray jobsArray = new JSONArray(jobsText);
        List<Job> jobs = new ArrayList<>();

        for (int i = 0; i < jobsArray.length(); i++) {
            JSONObject j = jobsArray.getJSONObject(i);

            Job job = new Job();
            job.datasetSize = j.getInt("dataset_size");
            job.nbTasks = j.getInt("nb_tasks");
            job.taskDuration = j.getInt("task_duration");

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
        // Generate infra
        //List<NodeConfig> nodes = generateHeterogeneousInfrastructureEquilibre(config);
        List<NodeConfig> nodes = loadNodeConfigCSV("/Users/cherif/Documents/Traveaux/CSPModel/results-using-CSP/offline-mode-arriving-50jobs/nodes_config.csv");
        int[] bandwidths = new int[nodes.size()];
        double[] cpus = new double[nodes.size()];

        for (int i = 0; i < nodes.size(); i++) {
            NodeConfig node = nodes.get(i);
            bandwidths[i] = node.bandwidth;
            cpus[i] = node.computationNodes;
        }

        int[] starting_times = new int[nodes.size()];
        if (config.sameStartingTime) {
            for (int i = 0; i < nodes.size(); i++) {
                starting_times[i] = 0;
                System.out.println("Job "+i+" Start at time: "+ starting_times[i]);
            }
        } else {
            double s = 0;
            //starting_times = new int[]{40, 41, 54, 64, 118, 163, 252, 256, 277, 279, 289, 317, 318, 327, 369, 400, 410, 446, 512, 512};
            starting_times = new int[]{40, 41, 54, 64, 118, 163, 252, 256, 277, 279, 289, 317, 318, 327, 369, 400, 410, 446, 512, 512, 578, 626, 642, 649, 775, 792, 795, 800, 875, 912, 978, 1030, 1061, 1205, 1224, 1256, 1327, 1366, 1445, 1479, 1528, 1530, 1540, 1554, 1557, 1568, 1572, 1585, 1626, 1644, 1662, 1672, 1684, 1794, 1836, 1874, 1881, 1934, 1941, 1960, 2142, 2183, 2216, 2262, 2336, 2396, 2406, 2407, 2422, 2435, 2444, 2559, 2643, 2658, 2700, 2720, 2819, 2843, 2856, 2867, 2900, 2912, 2947, 3039, 3059, 3069, 3309, 3338, 3341, 3343, 3348, 3387, 3450, 3472, 3475, 3494, 3716, 3746, 3888, 3967, 3967, 4018, 4064, 4095, 4107, 4148, 4153, 4176, 4200, 4323, 4407, 4419, 4447, 4454, 4552, 4634, 4648, 4689, 4726, 4733, 4790, 4821, 4882, 4912, 4912, 4928, 4928, 5034, 5119, 5190, 5205, 5207, 5291, 5409, 5412, 5439, 5442, 5499, 5557, 5562, 5588, 5620, 5632, 5715, 5737, 5746, 5777, 5830, 5839, 5854, 6067, 6109, 6132, 6161, 6166, 6176, 6193, 6228, 6239, 6249, 6252, 6292, 6302, 6396, 6475, 6478, 6489, 6533, 6542, 6548, 6658, 6692, 6717, 6779, 6845, 6853, 6857, 6880, 6902, 6927, 6979, 7024, 7190, 7194, 7214, 7231, 7310, 7322, 7330, 7354, 7376, 7389, 7400, 7503, 7526, 7605, 7637, 7640, 7929, 8001};
            for(int i=0;i<starting_times.length; i++) System.out.println("Job "+i+" Start at time: "+starting_times[i]);

            /*for (int i = 0; i < nbData; i++) {
                s += -Math.log(1 - random.nextDouble()) * config.lambdaRate;
                starting_times[i] = (int) s;
                System.out.println("Job "+i+" Start at time: "+ starting_times[i]);
            }*/
        }

        Scheduling sch = new Scheduling();
        // Call scheduler
        Scheduling.SchedulingResult result = Scheduling.runScheduler(nodes.size(), nbData, data_sizes, works, bandwidths, cpus, starting_times);

        String basePath = "/Users/cherif/Documents/Traveaux/CSPModel/results-using-CSP/offline-mode-arriving-200jobs";

        // ---- Pure Java JSON saving ----
        String worksJson = toJsonArray(result.worksExec);
        String transfersJson = toJsonArray(result.transfers);

        //writeTextFile(basePath + "/works_exec_solution.json", worksJson);
        //writeTextFile(basePath + "/transfers_solution.json", transfersJson);

        Scheduling.writeNodeConfigCSV(nodes, basePath + "/nodes.csv");
        Scheduling.writeTransferConfigCSV(result.transfers, basePath + "/transfers_solution.csv");
        writeWorkConfigCSV(result.worksExec, basePath + "/works_exec_solution.csv");

        logger.info("Terminé.");
    }
}
