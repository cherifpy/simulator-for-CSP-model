package simulator.utils.CSPModel;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Genetic algorithm scheduler adapted from the CSP model you provided.
 *
 * - Single data item (like your example).
 * - Random nodes (bandwidth, cpu) and random works per data.
 * - Chromosome: for each work (flattened across data), an int node index.
 * - Fitness: makespan computed by scheduling transfer (start=0) and works sequentially per node.
 *
 * Usage: compile and run. Console output only.
 */
public class GeneticScheduler {

    // ====== problem data (kept similar to your original Java/CSP) ======
    static final int CPU_UNIT = 1;
    static final int NB_NODES = 20;
    static final int NB_DATA = 1;
    static final int NB_MIN_WORKS = 1;
    static final int NB_MAX_WORKS = 20;
    static final int MAX_DURATION_PER_WORK = 90; // not directly used here (we set durations as 10 or 100)
    static final long RANDOM_SEED = 0L; // same approach as your code (Random(0))

    // ====== GA parameters ======
    static final int POP_SIZE = 200;
    static final int GENERATIONS = 500;
    static final double CROSSOVER_RATE = 0.9;
    static final double MUTATION_RATE = 0.08;
    static final int TOURNAMENT_SIZE = 3;
    static final int ELITISM = 2; // keep top 2 each generation
    static final boolean VERBOSE = true;

    static final Random rnd = new Random(0);

    // Problem instance structures
    static int[] dataSizes;                // size per data (MB)
    static List<List<Integer>> worksByData; // works weights per data (e.g., [100,100,...])
    static int[] bandwidths;               // MB/s per node
    static int[] cpus;                     // units/s per node

    // Flattened mapping of work index: we flatten works for all data into a single list
    static class WorkInfo {
        int dataIndex;
        int workIndexInData;
        int weight; // e.g., 10 or 100
        WorkInfo(int d, int widx, int weight) { this.dataIndex = d; this.workIndexInData = widx; this.weight = weight; }
    }
    static List<WorkInfo> flatWorks;

    // Individual (chromosome)
    static class Individual {
        int[] assignment; // assignment[workId] = nodeIndex
        double fitness;   // lower is better (makespan)
        int makespanInt;  // cached integer makespan for printing
        Individual(int nWorks) {
            this.assignment = new int[nWorks];
            this.fitness = Double.POSITIVE_INFINITY;
            this.makespanInt = Integer.MAX_VALUE;
        }
        Individual cloneIndividual() {
            Individual c = new Individual(this.assignment.length);
            System.arraycopy(this.assignment, 0, c.assignment, 0, this.assignment.length);
            c.fitness = this.fitness;
            c.makespanInt = this.makespanInt;
            return c;
        }
    }

    public static void main(String[] args) {
        // build problem instance
        buildRandomInstance();

        // flatten works
        buildFlatWorks();

        // run GA
        Individual best = runGA();

        // print best solution
        printSolution(best, "FINAL BEST");
    }

    static void buildRandomInstance() {
        dataSizes = new int[NB_DATA];
        worksByData = new ArrayList<>();
        // data sizes: choose 2048 or 40960 (as in your original)
        for (int i = 0; i < NB_DATA; i++) {
            dataSizes[i] = 37315; // rnd.nextBoolean() ? 2048 : 40960;
        }
        // works: choose between 10 and 100 repeated nbWorks times
        for (int i = 0; i < NB_DATA; i++) {
            int nbWorksForData = rnd.nextInt(NB_MAX_WORKS - NB_MIN_WORKS + 1) + NB_MIN_WORKS;
            boolean pick10 = rnd.nextBoolean();
            List<Integer> list = new ArrayList<>();
            for (int k = 0; k < 4; k++) list.add(58);
            worksByData.add(list);
        }

        // nodes
        bandwidths = new int[NB_NODES];
        cpus = new int[NB_NODES];
        for (int j = 0; j < NB_NODES; j++) {
            bandwidths[j] = rnd.nextInt(800 - 12 + 1) + 12; // [12,800]
            cpus[j] = (rnd.nextInt(8) + 1) * CPU_UNIT;     // 1..8 * CPU_UNIT
        }

        // print summary
        System.out.println("=== PROBLEM INSTANCE ===");
        for (int i = 0; i < NB_DATA; i++) {
            System.out.println(" Data " + i + ": size=" + dataSizes[i] + " MB, works=" + worksByData.get(i));
        }
        System.out.println(" NODES");
        for (int j = 0; j < NB_NODES; j++) {
            System.out.println("  Node " + j + ": bandwidth=" + bandwidths[j] + " MB/s, cpu=" + cpus[j] + " units/s");
        }
        System.out.println("========================\n");
    }

    static void buildFlatWorks() {
        flatWorks = new ArrayList<>();
        for (int i = 0; i < NB_DATA; i++) {
            List<Integer> wl = worksByData.get(i);
            for (int k = 0; k < wl.size(); k++) {
                flatWorks.add(new WorkInfo(i, k, wl.get(k)));
            }
        }
    }

    // GA main loop
    static Individual runGA() {
        int nWorks = flatWorks.size();
        // init population
        Individual[] pop = new Individual[POP_SIZE];
        for (int p = 0; p < POP_SIZE; p++) {
            pop[p] = randomIndividual(nWorks);
            evaluate(pop[p]);
        }
        // sort by fitness ascending (lower makespan is better)
        Arrays.sort(pop, Comparator.comparingDouble(ind -> ind.fitness));

        Individual bestSoFar = pop[0].cloneIndividual();

        for (int gen = 0; gen < GENERATIONS; gen++) {
            Individual[] next = new Individual[POP_SIZE];

            // elitism: copy best ELITISM individuals
            for (int e = 0; e < ELITISM; e++) next[e] = pop[e].cloneIndividual();

            // fill rest by selection/crossover/mutation
            int idx = ELITISM;
            while (idx < POP_SIZE) {
                Individual parent1 = tournament(pop);
                Individual parent2 = tournament(pop);

                Individual child;
                if (rnd.nextDouble() < CROSSOVER_RATE) child = crossover(parent1, parent2);
                else child = parent1.cloneIndividual();

                mutate(child);

                evaluate(child);
                next[idx++] = child;
            }

            // replace
            pop = next;
            Arrays.sort(pop, Comparator.comparingDouble(ind -> ind.fitness));

            if (pop[0].fitness < bestSoFar.fitness) {
                bestSoFar = pop[0].cloneIndividual();
            }

            if (VERBOSE && (gen % 25 == 0 || gen == GENERATIONS-1)) {
                System.out.printf("Gen %4d : best makespan = %d\n", gen, pop[0].makespanInt);
            }
        }

        return bestSoFar;
    }

    // Create a random individual
    static Individual randomIndividual(int nWorks) {
        Individual ind = new Individual(nWorks);
        for (int w = 0; w < nWorks; w++) {
            ind.assignment[w] = rnd.nextInt(NB_NODES); // random node
        }
        return ind;
    }

    // Tournament selection
    static Individual tournament(Individual[] pop) {
        Individual best = null;
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Individual cand = pop[rnd.nextInt(pop.length)];
            if (best == null || cand.fitness < best.fitness) best = cand;
        }
        return best;
    }

    // One-point crossover
    static Individual crossover(Individual a, Individual b) {
        int n = a.assignment.length;
        Individual child = new Individual(n);
        int cut = rnd.nextInt(Math.max(1, n)); // cut in [0,n-1] effectively (if n==1 cut=0)
        for (int i = 0; i < n; i++) {
            child.assignment[i] = (i <= cut) ? a.assignment[i] : b.assignment[i];
        }
        return child;
    }

    // Mutation: random reset of some genes
    static void mutate(Individual ind) {
        for (int i = 0; i < ind.assignment.length; i++) {
            if (rnd.nextDouble() < MUTATION_RATE) {
                ind.assignment[i] = rnd.nextInt(NB_NODES);
            }
        }
    }

    // Evaluate makespan (fitness) of an individual
    static void evaluate(Individual ind) {
        // Map node -> list of works
        List<Integer>[] nodeWorks = new ArrayList[NB_NODES];
        for (int j = 0; j < NB_NODES; j++) nodeWorks[j] = new ArrayList<>();
        for (int w = 0; w < ind.assignment.length; w++) {
            int node = ind.assignment[w];
            if (node < 0 || node >= NB_NODES) node = 0;
            nodeWorks[node].add(w);
        }

        int globalMakespan = 0;

        for (int j = 0; j < NB_NODES; j++) {
            List<Integer> assigned = nodeWorks[j];
            if (assigned.isEmpty()) continue;

            int time = 0;
            Set<Integer> transferredData = new HashSet<>();

            // Sort works by data index (generic for multiple data)
            assigned.sort(Comparator.comparingInt(w -> flatWorks.get(w).dataIndex));

            for (int w : assigned) {
                WorkInfo wi = flatWorks.get(w);
                int dataIndex = wi.dataIndex;

                // Only transfer once per node per data
                if (!transferredData.contains(dataIndex)) {
                    int dataSize = dataSizes[dataIndex];
                    int transferDur = (int) Math.ceil((double) dataSize / Math.max(1, bandwidths[j]));
                    time += transferDur;
                    transferredData.add(dataIndex);
                }

                // Work duration
                int workDur = (int) Math.ceil((double) wi.weight / Math.max(1, cpus[j]));
                time += workDur;
            }

            if (time > globalMakespan) globalMakespan = time;
        }

        ind.makespanInt = globalMakespan;
        ind.fitness = (double) globalMakespan;
    }

    // Print readable solution (transfers and per-node work schedule summary)
    static void printSolution(Individual ind, String title) {
        System.out.println("\n=== " + title + " SOLUTION ===");
        System.out.println("makespan = " + ind.makespanInt);
        // Build per-node assigned works
        List<List<Integer>> assignedWorksPerNode = new ArrayList<>();
        for (int j = 0; j < NB_NODES; j++) assignedWorksPerNode.add(new ArrayList<>());
        for (int w = 0; w < ind.assignment.length; w++) {
            int node = ind.assignment[w];
            assignedWorksPerNode.get(node).add(w);
        }

        for (int j = 0; j < NB_NODES; j++) {
            System.err.println("\nNode " + j + " schedule:");
            List<Integer> assigned = assignedWorksPerNode.get(j);
            if (assigned.isEmpty()) continue;

            // Track which data items are transferred to this node
            Set<Integer> transferredData = new HashSet<>();
            int time = 0;

            // Sort works by data index (so we can print transfers first)
            assigned.sort(Comparator.comparingInt(w -> flatWorks.get(w).dataIndex));

            // First, print transfers
            for (int w : assigned) {
                WorkInfo wi = flatWorks.get(w);
                int dataIndex = wi.dataIndex;
                if (!transferredData.contains(dataIndex)) {
                    int dataSize = dataSizes[dataIndex];
                    int transferDur = (int) Math.ceil((double) dataSize / Math.max(1, bandwidths[j]));
                    System.out.println("  Transfer d" + dataIndex + " dur=" + transferDur + " (start=" + time + ", end=" + (time + transferDur) + ")");
                    time += transferDur;
                    transferredData.add(dataIndex);
                }
            }

            // Now schedule works on this node
            List<int[]> workPairs = new ArrayList<>();
            for (int w : assigned) {
                int dur = (int) Math.ceil((double) flatWorks.get(w).weight / Math.max(1, cpus[j]));
                workPairs.add(new int[]{w, dur});
            }

            // Optional: sort works descending by duration (LPT heuristic)
            workPairs.sort((a, b) -> Integer.compare(b[1], a[1]));

            for (int[] p : workPairs) {
                int w = p[0], dur = p[1];
                int start = time;
                int end = start + dur;
                System.out.println("  Work " + flatWorks.get(w).workIndexInData + " of data " + flatWorks.get(w).dataIndex
                        + " (globalWorkId=" + w + ", w=" + flatWorks.get(w).weight + ") processed from " + start + " to " + end + " (dur=" + dur + ")");
                time = end;
            }
        }   
        System.out.println("========================\n");

    }
}
