Here is a plan. I'm not entering plan mode formally since you asked for a description to react to.                                                                                                    

1. Read of the current state                                                                                                                                                                          

src/cellbox (~1700 LOC) carries a lot of machinery that none of the new TF-driven work needs:                                                                                                         
- BaseEstimator (325 LOC) + CellBoxEstimator (410 LOC) — abstract estimator, save/load via orbax, periodic perturbation validation, TB logging, dataloader iteration, train/val splits.               
- 4 Flax modules — CellBoxSteadyState{,DS,NB,NBDS} — all built around iterative fixed-point rollout over a gene×gene matrix A.                                                                        
- RegulatorNet (DeepSets edge-based aggregation).                                                                                                                                                     
- metrics.py — generic LFC profile metrics (the only piece worth pulling forward as-is).                                                                                                              
- predict_from_tfs.py — a thin script that reuses CellBoxEstimator.predict to score predictions when U is one-hot on the perturbation, then writes an AnnData of predictions. Not aligned with the    
notebook's actual TF-input model.                                                                                                                                                                     

The notebook's model is structurally different from anything in src/cellbox: input is (B, n_tfs) of TF expression, not a one-hot perturbation, and there is no fixed-point rollout. Trying to reuse   
BaseEstimator would mean either bending U semantics or adding another model_type branch — both push complexity into legacy code that wants to be frozen.                                              
                                                        
2. Archival proposal — minimal-friction                                                                                                                                                               
                                                        
src/legacy/                                                                                                                                                                                           
├── __init__.py          # empty                          
├── cellbox/             # ← src/cellbox moves here untouched                                                                                                                                         
│   └── ...
└── evaluation/          # ← src/evaluation moves here untouched                                                                                                                                      
    └── evaluate.py      # rewrite one import: cellbox.metrics → legacy.cellbox.metrics                                                                                                               

Because the new package is tf_prediction, no symbol overlap is possible (no CellBox*, no BaseEstimator, no RegulatorNet in the new namespace). The "naming overlap" concern resolves itself by choice 
of package name — you don't need rename shims.                                                                                                                                                        

Migration steps:                                                                                                                                                                                      
1. git mv src/cellbox src/legacy/cellbox + git mv src/evaluation src/legacy/evaluation.
2. Touch src/legacy/__init__.py (empty).                                                                                                                                                              
3. Patch the one cross-package import in evaluate.py (from cellbox.metrics → from legacy.cellbox.metrics).
4. Update any experiment script / config that did from cellbox import … to from legacy.cellbox import …. Grep should find all of them in experiments/06052026_cellbox_noise/.                         
5. Old notebooks still work since they sys.path.insert(0, "/workspace/src") and now import legacy.cellbox explicitly.                                                                                 

No code rewrites inside legacy/cellbox/ itself — the relative imports (from .base_estimator, etc.) keep working.                                                                                      

3. New package — src/tf_prediction/                                                                                                                                                                   

Scope: predict per-gene NB counts from a TF-expression vector, with an optional regulator mask. No rollout. No save/load yet (add only when an experiment actually needs it).                         
                                                        
src/tf_prediction/                                                                                                                                                                                    
├── __init__.py          # re-export the public surface   
├── data.py              # adata → (X_tf, Y_raw, lib, tf_genes, ctrl_lcp_mean, tf_mu, tf_sigma)                                                                                                       
├── regulondb.py         # build_tf_mask(var_names, ref_db) → (tf_genes, Amask_tf)                                                                                                                    
├── models.py            # TFLinearNB, BaselineConstantNB  (both nn.Module, same call sig)
├── train.py             # fit(model, X, Y, lib, *, n_epochs, batch_size, lr, key) → (state, history)                                                                                                 
├── eval.py              # per_gene_nll, per_perturbation_nll, fit_baseline_overdispersion                                                                                                            
└── metrics.py           # copy of legacy.cellbox.metrics.profile_metrics  (67 LOC, no churn)                                                                                                         

Recommended architecture — functional, not OO                                                                                                                                                         

For this scope I'd recommend skipping the scvi-tools-style estimator and going functional. The reasons:                                                                                               
                                                        
- No shared training loop yet. scvi-tools' value is OO when you have many models that share a complex fit loop with hooks, callbacks, periodic validation. Your fit loop here is ~30 lines. An        
Estimator base class would carry more lines than the loops it factors out.
- JAX/Flax idioms are already functional. Models are nn.Module (data-class-like), but state lives in TrainState. A fit(model, X, Y, lib, hp) → state function is the natural shape; the model class   
doesn't own state.                                                                                                                                                                                    
- Composability for diagnostics. Most of the notebook is per-gene-NLL slicing. Functions that take (model, params, batch) → per_gene_nll compose trivially: you can swap models (causal vs
unconstrained vs baseline) at the call site without sub-classing.                                                                                                                                     
- Math stays visible. With four short files the entire forward pass and likelihood fits on one screen — your "mathematical maturity" criterion.

Concretely the public surface ends up looking like:       

from tf_prediction import (                               
    prepare_tf_arrays,        # adata + ref_db → all jnp arrays
    TFLinearNB,                                                                                                                                                                                       
    BaselineConstantNB,
    fit,                      # fit(model, X, Y, lib, ...) → (state, history)                                                                                                                         
    per_gene_nll,             # (model, params, X, Y, lib) → (n_genes,)
    per_perturbation_nll,     # (model, params, X, Y, lib, pert_idx) → (n_perts, n_genes)                                                                                                             
    fit_baseline_overdispersion,                                                                                                                                                                      
) 

Notebook then becomes ~80 lines of plotting on top of these primitives.                                                                                                                               

Model class: one module, mask is optional                                                                                                                                                             
                                                        
Keep the TFLinearNB shape you already have. The baseline becomes a sibling BaselineConstantNB(nn.Module) with the same __call__(x_tf, y_raw, lib) signature (it just ignores x_tf). That way          
per_gene_nll, per_perturbation_nll, and fit work on all three (unconstrained / causal / baseline) without branching.

Open question: keep TFLinearNB as the only model, or factor a TFNonlinearNB (MLP head) at the same time? I'd hold off — YAGNI — until you actually need it.                                           

What does not move                                                                                                                                                                                    
                                                        
- RegulatorNet, CellBoxSteadyState* modules, rollout training step → stay in legacy. Untouched.                                                                                                       
- BaseEstimator.save/load via orbax → not ported. If you need checkpointing later, a 10-line jnp.save(params) is fine for now.
- _get_predict_dataloader / _predict_batch plumbing → not ported. Direct batched calls over jnp.array(X[s:s+bs]).                                                                                     

4. Order of operations                                                                                                                                                                                

1. Decide on the package skeleton above (or push back on parts).                                                                                                                                      
2. Move legacy (Phase 1, ~5 min, fully mechanical).
3. Create tf_prediction/ and port the notebook's logic in this order: regulondb.py, data.py, models.py, train.py, eval.py. Each file should be small enough to read top-to-bottom.                    
4. Rewrite tf_causal_regression.ipynb so cells 1–3 just call the package; cells 4+ keep the figures.                                                                                                  
5. Smoke test: same numbers on the lce75 subset as the current notebook (val NLL baseline / unconstrained / causal). This is the ground truth that proves no regression during refactor.              

5. Things I'd flag                                                                                                                                                                                    

- profile_metrics duplication. Copying 67 lines into tf_prediction/metrics.py is cleaner than from legacy.cellbox.metrics import … (which puts a forward dependency from new code into archived code).
The duplication is fine; it's a stable utility.
- adata preprocessing. The notebook's MIN_LIB, EXPERIMENT_SUBSET, lowercasing, layer adds — these are dataset-conditioning steps, not modelling. Put them in data.py with explicit kwargs so they're  
parameterizable, don't bake the lce75 path in.                                                                                                                                                        
- pgi_kd.py and tf_causal_zscores.ipynb in the same dir look related; if they share data prep, they should consume the new package too, so plan for that even though we're not porting them in this
round.                                                                                                                                                                                                
                                                        
Want me to proceed with the archive move (Phase 1) so you can review that as a discrete step before I touch the new package?