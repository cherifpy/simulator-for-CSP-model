
import subprocess

def schedulingUsingJavaCSP():
    """
    Wrapper to call the Java CSP solver via command line.
    """
    
    result = subprocess.run(
        [
            "javac", 
            "-cp",
            "/csimohammed/javaCSP/Flowtime\ 2/lib/*",
            "-d",
            "/csimohammed/javaCSP/Flowtime\ 2/bin",
            "/csimohammed/javaCSP/Flowtime\ 2/src/Main.java"
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
            "/csimohammed/javaCSP/Flowtime\ 2/bin:/csimohammed/javaCSP/Flowtime\ 2/lib/*",    
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



if __name__ == "__main__":    
    model_output_path = "/csimohammed/javaCSP/Flowtime\ 2/outputs"
    schedulingUsingJavaCSP()
    "javac -cp /csimohammed/javaCSP/Flowtime\ 2/lib/* -d /csimohammed/javaCSP/Flowtime\ 2/bin /csimohammed/javaCSP/Flowtime\ 2/src/Main.java"


#    "java -cp /csimohammed/javaCSP/Flowtime\ 2/bin:/csimohammed/javaCSP/Flowtime\ 2/lib/* main.Main