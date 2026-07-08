import os
import json

def convert_jsonl_to_json(src_path, dest_path):
    if not os.path.exists(src_path):
        print(f" -> [ERROR] Source file not found: {src_path}")
        return False
    
    records = []
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f" -> Converted and saved: {dest_path} (Contains {len(records)} records)")
        return True
    except Exception as e:
        print(f" -> [ERROR] Failed to convert {src_path} to {dest_path}: {e}")
        return False

def populate_local_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data", "eval")
    os.makedirs(output_dir, exist_ok=True)
    
    cloned_repo_dir = os.path.join(base_dir, "FailureSensorIQ")
    if not os.path.exists(cloned_repo_dir):
        print(f"[ERROR] Cloned repository directory {cloned_repo_dir} not found. Please clone the repository first.")
        return
        
    fmsr_processed_dir = os.path.join(cloned_repo_dir, "eval_data", "fmsr_processed")
    eval_data_dir = os.path.join(cloned_repo_dir, "eval_data")
    
    mapping = {
        # output file name -> source file path in the cloned repository
        "failuresensoriq_single.json": os.path.join(fmsr_processed_dir, "filtered_data_all_Mar_30_2025.jsonl"),
        "failuresensoriq_multi.json": os.path.join(fmsr_processed_dir, "fmsr_filtered_mcmt_all.jsonl"),
        
        "failuresensoriq_standard_all.json": os.path.join(fmsr_processed_dir, "filtered_data_all_Mar_30_2025.jsonl"),
        "failuresensoriq_standard_all_multi_answers.json": os.path.join(fmsr_processed_dir, "fmsr_filtered_mcmt_all.jsonl"),
        "failuresensoriq_standard_all_10_options.json": os.path.join(fmsr_processed_dir, "filtered_data_all_Mar_30_2025_10options.jsonl"),
        "failuresensoriq_standard_sample_50.json": os.path.join(eval_data_dir, "industrial_mcp_original.jsonl"),
        
        "failuresensoriq_perturbed_simple.json": os.path.join(fmsr_processed_dir, "fmsr_filtered_perturbed_data_all_simple.jsonl"),
        "failuresensoriq_perturbed_complex.json": os.path.join(fmsr_processed_dir, "fmsr_filtered_perturbed_data_all_llama.jsonl"),
        "failuresensoriq_perturbed_10_options_simple.json": os.path.join(fmsr_processed_dir, "filtered_data_all_Mar_30_2025_10options_all_simple.jsonl"),
        "failuresensoriq_perturbed_10_options_complex.json": os.path.join(fmsr_processed_dir, "filtered_data_all_Mar_30_2025_10options_all_complex.jsonl"),
    }
    
    print("\nProcessing and converting FailureSensorIQ local files from cloned repository...")
    for dest_file, src_path in mapping.items():
        dest_path = os.path.join(output_dir, dest_file)
        convert_jsonl_to_json(src_path, dest_path)

if __name__ == "__main__":
    populate_local_data()
    print("\nLocal dataset preparation completed successfully!")
