
```
git config --global user.email "pierre.boyeau@gmail.com"
git config --global user.name "Pierre Boyeau"
```


```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


Based on literature, are any of these gene pairs verified TF-gene pairs.
If so, precise which one is a TF and precise if it is an activator or a repressor:


[I 2025-10-15 23:41:03,453] Trial 49 finished with value: 6.088644681767619e-07 and parameters: {'lambda_prior': 0.00040622493220879524, 'learning_rate': 0.04295825126547938, 'model_class': 'dynamic_hardmultiplicative'}. Best is trial 22 with value: 3.7231538385640306e-07.
Best trial:
  Value (minimal validation loss): 3.7231538385640306e-07
  Best hyperparameters: 
    lambda_prior: 0.016414183992245375
    learning_rate: 0.007941849105679976
    model_class: dynamic_multiplicative


Best trial:
  Value (maximal PRAUC): 0.04021650784776238
  Best hyperparameters: 
    lambda_prior: 4.986904864296812e-07
    learning_rate: 0.0016201885401608317
    n_epochs: 518
    model_class: dynamic_cellbox



- n_obs
- t


[I 2025-10-27 22:20:07,779] Trial 24 finished with value: 0.0051771042552216555 and parameters: {'learning_rate': 0.0002468672465397824, 'n_latent': 79, 'n_epochs': 4464}. Best is trial 8 with value: 0.005234284259723122.
Current value: 0.0051771042552216555, Current params: {'learning_rate': 0.0002468672465397824, 'n_latent': 79, 'n_epochs': 4464}
Best value: 0.005234284259723122, Best params: {'learning_rate': 0.00013521379657376357, 'n_latent': 152, 'n_epochs': 1163}
Best trial:
  Value (maximal PRAUC): 0.005234284259723122
  Best hyperparameters: 
    learning_rate: 0.00013521379657376357
    n_latent: 152
    n_epochs: 1163


Best alue:v 0.02069661351252447, Best params: {'lambda_prior': 3.8110191959621016e-05, 'learning_rate': 0.009979969893300964, 'model_class': 'dynamic_cellbox'}
Best trial:
  Value (maximal PRAUC): 0.02069661351252447
  Best hyperparameters: 
    lambda_prior: 3.8110191959621016e-05
    learning_rate: 0.009979969893300964
    model_class: dynamic_cellbox


1. compare two best models
- study their correlation, prauc, discoveries, etc.

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/ode_experiment_10282025"
python ode_script.py \
--config=../src/essential/configs/models/dynamic_cellbox.py \
--output_path $OUTPUT_PATH/

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/ode_experiment_10282025"
python ode_script.py \
--config=../src/essential/configs/models/dynamic_cellboxlowdim2.py \
--output_path $OUTPUT_PATH

2. find the best model, compare one batch vs all batches

