import yaml
import pandas as pd

configfile: "config.yaml"

TARGET_KERNELS = []
methods_df = pd.DataFrame(config["methods"])

for task_name in config["tasks"].keys():
    for _, row in methods_df.iterrows():
        method = row["method"]
        config_id = row["config_id"]
        target_path = f"results/{task_name}/{method}/{config_id}/kernel.csv"
        TARGET_KERNELS.append(target_path)

rule all:
    input:
        TARGET_KERNELS

rule write_params:
    output:
        "results/{task}/{method}/{config_id}/params.json"
    run:
        import json
        import pandas as pd
        
        methods_df = pd.DataFrame(config["methods"])
        # Query the exact params for this wildcard combination
        match = methods_df[
            (methods_df["method"] == wildcards.method) & 
            (methods_df["config_id"] == wildcards.config_id)
        ]
        
        params = match.iloc[0]["params"]
        
        with open(output[0], 'w') as f:
            json.dump(params, f)

rule generate_representation:
    input:
        genes=lambda wildcards: config["tasks"][wildcards.task]["genes_csv"],
        params="results/{task}/{method}/{config_id}/params.json"
    output:
        kernel="results/{task}/{method}/{config_id}/kernel.csv",
        expectations="results/{task}/{method}/{config_id}/expectations.csv",
        fluxes="results/{task}/{method}/{config_id}/flux_cache.csv"
    params:
        model_type=lambda wildcards: config["tasks"][wildcards.task]["model_type"]
    threads: 8
    shell:
        """
        python scripts/generate_representations.py \
            --method {wildcards.method} \
            --model_type {params.model_type} \
            --genes_csv {input.genes} \
            --config_json {input.params} \
            --out_kernel {output.kernel} \
            --out_expectations {output.expectations} \
            --threads {threads} \
            --cache_fluxes {output.fluxes}
        """