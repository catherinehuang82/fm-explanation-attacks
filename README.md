# On the Privacy Risks of Post-Hoc Explanations of Foundation Models
Code for the paper _On the Privacy Risks of Post-Hoc Explanations of Foundation Models_. To be presented at the ICML 2024 Workshop on Foundation Models in the Wild.

---

### Abstract
Foundation models are becoming increasingly deployed in high-stakes contexts in fields such as medicine, finance, and law. In these contexts, there is a trade-off between model _explainability_ and data _privacy_: explainability promotes transparency, and privacy is a limit on transparency. In this work, we push the boundaries of this trade-off: we reveal that post-hoc feature attribution explanations beget unforeseen privacy risks upon the fine-tuning data of vision transformer models. We construct VAR-LRT and L1/L2-LRT, two new membership inference attacks leveraging feature attribution explanations that are significantly more successful than existing explanation-leveraging attacks, particularly in the low false-positive rate regime that allows an adversary to identify specific fine-tuning dataset members with high confidence. We carry out a systematic empirical investigation of our 2 new attacks with 5 vision transformer architectures, 5 benchmark datasets, and 4 state-of-the-art post-hoc explanation methods. Our work addresses the lack of trust in post-hoc explanation methods that has contributed to the slow adoption of foundation models in high-stakes domains.

---
### Package Requirements
* Python >= 3.8
* PyTorch >= 2.0
* Captum (`conda install captum -c pytorch` or `conda install captum -c conda-forge`
* timm (`pip install timm`) for PyTorch image models

---
### Running Experiments
The scripts, as they are configured, generate attack metrics and (log-scaled and linearly scaled) ROC curves for one singular setting (of dataset, model, explanation type, and attack type). Run the scripts, in the following order, to run the pipeline:
* `sbatch scripts/train_driver.sh` to fine-tune models and save their state dictionaries
* `sbatch scripts/get_explanations_driver.sh` to compute post-hoc explanations and save per-example attack scores: explanation variance, L1 norm, L2 norm
* `python3 run_attack.py` to run the attack and generate metrics + plots

---
### Code Organization
This repository is organized as follows:
* `train.py` is the script for fine-tuning a single vision transformer model and saving its state dictionary.
* `get_explanations.py` is the script that, for a single model, computes post-hoc explanations on all data examples and saves per-example attack scores: explanation variance, L1 norm, L2 norm.
* `run_attack.py` is the script that, for a single attack parameter setting, runs the attack and generates metrics and plots.
* `get_losses.py` is the script that, for a single model, computes per-example cross-entropy losses for the Loss LiRA baseline.
* `run_losses_attack.py` is the script that, for a single attack parameter setting, runs Loss LiRA.
* `attack_metrics.ipynb` is a helper notebook that generates tables that succinctly display AUC and TPR @ FPR = 0.01, 0.001 for a particular inputted attack setting.
* `scripts/` holds experiment bash scripts. As of right now, the scripts must be run using the `sbatch` Slurm command, but we are working towards having at least some scripts be runnable with `bash`.
* `attack_data/` holds data helpful in the attack pipeline, from model state dicts to explanation scores, as well as any reported metrics, such as model accuracies and attack TPRs at certain FPRs.
* `data/` holds the datasets [CIFAR-10, CIFAR-100, GTSRB, SVHN, Food 101] downloaded off of `torchvision.datasets` but is empty in this repository.

