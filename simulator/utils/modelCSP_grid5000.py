
import subprocess

def schedulingUsingJavaCSP():
    """
    Wrapper to call the Java CSP solver via command line.
    """
    
    result = subprocess.run(
        [
            "javac", 
            "-cp",
            #/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/lib
            "/csimohammed/simulator-for-CSP-model/simulator/utils/model/lib/*",
            "-d",
            "/csimohammed/simulator-for-CSP-model/simulator/utils/model/bin",
            "/csimohammed/simulator-for-CSP-model/simulator/utils/model/src/Main.java"
        ],
        capture_output=True,
        text=True
    )  
    
    print("Compilation Error")
    print(str(result.stderr))
    print("results of compilation")
    print(str(result.stdout))


    result = subprocess.run(
        [
            "java", 
            "-cp",
            "/csimohammed/simulator-for-CSP-model/simulator/utils/model/lib/*",    
            "main.Main"
        ],
        capture_output=True,
        text=True
    )  

    print("results of execution")
    print(str(result.stderr))

    transfers = {}
    works = {}

    return transfers, works # Implementation would go here



def getResults(jobs, master_node, nb_data, nb_nodes, nb_works, output_path: str):
    import json

    with open(output_path, "r") as f:
        data = json.load(f)
    if len(data) == 0:
        return {}, {}
    transfers = {f"node_{j}": [] for j in range(nb_nodes)}
    works = {f"node_{j}": [] for j in range(nb_nodes)}

    for j in range(nb_nodes):
        for i in range(nb_data):
            for k in range(nb_works[i]):
                if data["work_height"][j][i][k] == 1:
                    works[f"node_{j}"].append((jobs[i].job_id, master_node.compute_nodes[j].node_id, k, data["work_start"][j][i][k], data["work_end"][j][i][k], data["work_end"][j][i][k] - data["work_start"][j][i][k]))
    
    for j in range(nb_nodes):
        for i in range(nb_data):
            #dict_info[f"node_{node_index}"].append((job_index, node_index, task_index, start_time, end_time, end_time - start_time ))
            if data["transfer_height"][j][i] == 1:
                transfers[f"node_{j}"].append((jobs[i].job_id, master_node.compute_nodes[j].node_id, data["transfer_start"][j][i], data["transfer_end"][j][i], data["transfer_end"][j][i] - data["transfer_start"][j][i]))

    
    return transfers, works


#if __name__ == "__main__":    
#    model_output_path = "/csimohammed/simulator-for-CSP-model/simulator/utils/model/outputs"
#    schedulingUsingJavaCSP()
#    "javac -cp /csimohammed/simulator-for-CSP-model/simulator/utils/model/lib/* -d /csimohammed/simulator-for-CSP-model/simulator/utils/model/bin /csimohammed/simulator-for-CSP-model/simulator/utils/model/src/Main.java"


#    "java -cp /csimohammed/javaCSP/Flowtime\ 2/bin:/csimohammed/javaCSP/Flowtime\ 2/lib/* main.Main