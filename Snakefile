import pandas as pd

configfile: "config.yaml"

TARGET_KERNELS = []
methods_df = pd.DataFrame(config["methods"])

for task_name, task_cfg in config["tasks"].items():
    # for _, row in methods_df.iterrows():
    #     method = row["method"]
    #     config_id = row["config_id"]
    #     target_path = f"results/{task_name}/{method}/{config_id}/kernel.pkl"
    #     TARGET_KERNELS.append(target_path)
        
    #     if "transcriptomic_adata" in task_cfg:
    #         for target_type in ["mean_pca", "mmd"]:
    #             TARGET_KERNELS.append(f"results/{task_name}/{method}/{config_id}/kernel_metrics_{target_type}.json")
    
    if "transcriptomic_adata" in task_cfg:
        for target_type in ["mean_pca", "mmd"]:
            TARGET_KERNELS.append(f"results/{task_name}/targets/{target_type}_distances.csv")
            TARGET_KERNELS.append(f"results/{task_name}/targets/{target_type}_kernel.pkl")

rule all:
    input:
        TARGET_KERNELS

rule write_params:
    input:
        cfg="config.yaml"
    output:
        "results/{task}/{method}/{config_id}/params.json"
    run:
        import json

        task_cfg = config["tasks"][wildcards.task]
        methods_df = pd.DataFrame(config["methods"])
        match = methods_df[
            (methods_df["method"] == wildcards.method) &
            (methods_df["config_id"] == wildcards.config_id)
        ]

        run_config = {
            "method": wildcards.method,
            "model_type": task_cfg["model_type"],
            "genes_csv": task_cfg["genes_csv"],
            "params": match.iloc[0]["params"],
        }

        with open(output[0], 'w') as f:
            json.dump(run_config, f, indent=2)

rule generate_representation:
    input:
        config_json="results/{task}/{method}/{config_id}/params.json"
    output:
        kernel="results/{task}/{method}/{config_id}/kernel.pkl",
        expectations="results/{task}/{method}/{config_id}/expectations.csv",
    params:
        cache_fluxes=lambda wildcards: f"results/{wildcards.task}/{wildcards.method}/{wildcards.config_id}/.flux_cache.csv"
    threads: 100
    shell:
        """
        python scripts/generate_representations.py \
            --config_json {input.config_json} \
            --out_kernel {output.kernel} \
            --out_expectations {output.expectations} \
            --threads 100 \
            --cache_fluxes {params.cache_fluxes}
        """

rule generate_targets_mean_pca:
    input:
        config="config.yaml"
    output:
        distances="results/{task}/targets/mean_pca_distances.csv",
        kernel="results/{task}/targets/mean_pca_kernel.pkl"
    shell:
        """
        python scripts/generate_targets_mean_pca.py \
            --task {wildcards.task} \
            --config {input.config} \
            --out_distances {output.distances} \
            --out_kernel {output.kernel}
        """

rule generate_targets_mmd:
    input:
        config="config.yaml"
    output:
        distances="results/{task}/targets/mmd_distances.csv",
        kernel="results/{task}/targets/mmd_kernel.pkl"
    shell:
        """
        python scripts/generate_targets_mmd.py \
            --task {wildcards.task} \
            --config {input.config} \
            --out_distances {output.distances} \
            --out_kernel {output.kernel}
        """

rule evaluate_kernel:
    input:
        pred_kernel="results/{task}/{method}/{config_id}/kernel.pkl",
        target_kernel="results/{task}/targets/{target_type}_kernel.pkl"
    output:
        metrics="results/{task}/{method}/{config_id}/kernel_metrics_{target_type}.json"
    shell:
        """
        python scripts/evaluate_kernel.py \
            --pred_kernel {input.pred_kernel} \
            --target_kernel {input.target_kernel} \
            --out_metrics {output.metrics} \
            --tag {wildcards.method}_{wildcards.config_id}_{wildcards.target_type}
        """
