# Discrete Diffusion for Biological Sequences: Reproducing and Extending SEDD

## Brief Summary

This project is a reproduction and research extension of [**Score-Entropy Discrete Diffusion (SEDD)**](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) for biological sequence generation. Starting from the official implementation, we adapt the framework to fine-tune on **biological sequence datasets** and study possible improvements for **discrete generative modeling** in scientific domains. In particular, the project investigates how SEDD can be transferred from general discrete data modeling to biologically meaningful token sequences, with emphasis on generation quality, training behavior, and task-specific modifications.

## Project Target

This project aims to bridge recent advances in discrete diffusion modeling with biological sequence generation. Specifically, it seeks to:

1. reproduce the original SEDD framework faithfully;
2. extend the method to biological sequence datasets through fine-tuning and domain adaptation;
3. analyze the suitability of discrete diffusion for modeling structured biological token distributions; and
4. identify practical improvements that enhance generation fidelity, optimization stability, and biological relevance.

## Why Discrete Diffusion?

Biological sequences are **naturally discrete objects**, formed by **tokens** such as DNA bases, RNA bases, or amino acids. Unlike continuous diffusion, discrete diffusion operates directly in token space, making it a more natural choice for sequence generation tasks.

## Environment Setup

To quickly set up the project environment, run:

```bash
git clone https://github.com/LiuYuqing14/Disrete_Diffusion_Entropy.git
cd Disrete_Diffusion_Entropy
conda env create -f environment.yml
conda activate sedd
```

## Training and Results

## 1. Human Genome hg38

<p align="center">
  <img src="image_src/hg38_sedd_pipeline_v2.svg" alt="SEDD pipeline for hg38 sequence modeling" width="900"/>
</p>
<p align="center">
  <sub>Training Pipeline for fine-tuning SEDD.</sub>
</p>

### Why hg38 is a good fit for discrete diffusion

A key advantage of discrete diffusion on genomic data is that the **corruption process remains biologically interpretable** throughout the entire generation trajectory.

For example, consider the following sequence corruption process:

```text
Complete sequence:  A T G C A T G C
Noise 25%:          A T G C ? T G C   <- still a valid fragment with a local gap
Noise 50%:          A ? G ? ? T G C   <- resembles a read with sequencing uncertainty
Noise 75%:          ? ? G ? ? ? G ?   <- only a few conserved positions remain
Full noise:         ? ? ? ? ? ? ? ?   <- completely unknown sequence
```
This makes the generation process itself resemble a realistic **biological inference** process: starting from a **fully unknown** sequence, the model gradually fills in each position according to `contextual dependency`, `local motif structure`, and `evolutionary conservation`, eventually reconstructing a meaningful **DNA segment**.
### Result

- **GC content:** `35% ± 5%`
- **N-base ratio:** `7%`
- **Sequence diversity:** partially repetitive

### Improvements
- Full set of **64 canonical 3-mers** from the DNA alphabet `{A, T, G, C}`; `N`-filtering is performed at the base level first, **before k-mer** conversion.
- For DNA sequences, `t ∈ [0.3, 0.7]` contains the **richest learning signal** for genomic reconstruction. A **cosine noise schedule** allocates more effective **masking probability** in this region.