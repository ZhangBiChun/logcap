import subprocess
import os
import sys

def run_ablation():
    """
    Runs an ablation study on the four data augmentation operators of LogCapMamba.
    Each run uses a different combination of operators and saves its own model cache and results.
    """
    
    # Base command arguments for THNS24 event level
    base_cmd = [
        sys.executable, "baselines.py", 
        "--selected_dataset", "THNS24", 
        "--selected_model", "LogCapMamba", 
        "--device", "cuda", 
        "--thns24_level", "event"
    ]
    
    # Define the ablation settings: (Run Name, Operators to keep)
    # 1: Semantic Shift, 2: Burst, 3: Off-hours, 4: Alien Fragment
    ablations = {
        "All_Ops": "1,2,3,4",
        "No_Semantic_Shift": "2,3,4",
        "No_Burst": "1,3,4",
        "No_Off_hours": "1,2,4",
        "No_Alien_Fragment": "1,2,3",
        "Only_Semantic_Shift": "1",
        "Only_Burst": "2",
        "Only_Off_hours": "3",
        "Only_Alien_Fragment": "4"
    }
    
    print("===============================================================")
    print("Starting LogCapMamba Data Augmentation Ablation Study")
    print("===============================================================\n")
    
    for tag, ops in ablations.items():
        print(f"===============================================================")
        print(f">>> Running Ablation: {tag}")
        print(f">>> Active Operators: [{ops}]")
        print(f"===============================================================")
        
        cmd = base_cmd + [
            "--active_aug_ops", ops,
            "--ablation_tag", tag
        ]
        
        try:
            # Run the command and stream the output to console
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                universal_newlines=True
            )
            process.wait()
            
            if process.returncode != 0:
                print(f"[-] Error: Ablation run {tag} failed with return code {process.returncode}")
            else:
                print(f"[+] Successfully completed run: {tag}\n")
                
        except Exception as e:
            print(f"[-] Exception occurred during run {tag}: {e}")
            break

    print("\n===============================================================")
    print("Ablation Study Completed!")
    print("Results are saved in: results/LogCapMamba_THNS24_summary.csv")
    print("===============================================================")

if __name__ == "__main__":
    run_ablation()
