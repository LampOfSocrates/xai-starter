# Protein Language Models — Lesson Notebooks

A hands-on path from "use a frozen pLM" to fine-tuning, interpretability, and
generation. Each lesson is an interactive notebook (`plm_l*.ipynb`) with the
explanations broken out as markdown. **Run the cells top to bottom** (*Run All*).

## Core lessons (1–5)

| # | Notebook | You'll learn | Task |
|---|----------|--------------|------|
| 1 | [Embeddings + probe](plm_l1_embeddings_probe.ipynb) | Frozen pLM as a feature extractor + sklearn head — the cheapest way to use a pLM | DeepSol solubility |
| 2 | [Zero-shot variants](plm_l2_zero_shot_variants.ipynb) | Score mutations with the masked-LM head, no training | Variant effect prediction |
| 3 | [Fine-tune classification](plm_l3_finetune_classification.ipynb) | End-to-end fine-tuning with HF `Trainer` | DeepSol solubility |
| 4 | [Token classification](plm_l4_token_classification.ipynb) | Per-residue prediction; label↔token alignment | Secondary structure |
| 5 | [Model comparison](plm_l5_model_comparison.ipynb) | Grid of models × pooling strategies; outputs CSV | Benchmark sweep |

## Advanced lessons (6–11)

| # | Notebook | You'll learn | Task |
|---|----------|--------------|------|
| 6 | [Attention → contacts](plm_l6_attention_contacts.ipynb) | ESM attention heads recover residue contacts, unsupervised (Rao 2020) | Attention-as-contact-predictor |
| 7 | [LoRA / PEFT](plm_l7_lora_peft.ipynb) | Parameter-efficient fine-tuning — train <1% of the weights | DeepSol solubility |
| 8 | [Structure-aware pLMs](plm_l8_structure_aware.ipynb) | The 3Di structural alphabet; ProstT5 / SaProt vs ESM-2 | Probe comparison |
| 9 | [Embedding geometry & retrieval](plm_l9_embedding_retrieval.ipynb) | PCA/t-SNE of embedding space; nearest-neighbour homology search | Family retrieval |
| 10 | [Generative design](plm_l10_inverse_folding.ipynb) | Masked-LM sampling; native-sequence recovery (sequence-only inverse folding) | Redesign a sequence |
| 11 | [Calibration & uncertainty](plm_l11_calibration.ipynb) | Reliability diagrams, ECE, temperature scaling | Calibrate a classifier |

Then continue to the [capstones](../capstones/README.md).

## Running

```bash
pip install -r requirements.txt
jupyter lab            # then open any plm_l*.ipynb and Run All
```

- Most lessons download a small ESM-2 model (`facebook/esm2_t6_8M_UR50D`, ~30 MB)
  and a dataset on first run; both cache locally.
- **L7** needs `peft` (`pip install peft`); **L8** uses `Rostlab/ProstT5`.

`plm_demo.ipynb` is the original starter and overlaps with L3 — safe to skip.

## Regenerating the notebooks

Generated from the scripts archived in
[../to_be_deleted/plm/](../to_be_deleted/plm/):

```bash
python _py_to_notebook.py to_be_deleted/plm/plm_l1_embeddings_probe.py
```
