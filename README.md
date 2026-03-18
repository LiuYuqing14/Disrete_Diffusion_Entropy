# Discrete Diffusion for Biological Sequences: Reproducing and Extending SEDD

## Brief Summary

This project is a reproduction and research extension of Score-Entropy Discrete Diffusion (SEDD) for biological sequence generation. Starting from the official implementation, we adapt the framework to fine-tune on biological sequence datasets and study possible improvements for discrete generative modeling in scientific domains. In particular, the project investigates how SEDD can be transferred from general discrete data modeling to biologically meaningful token sequences, with emphasis on generation quality, training behavior, and task-specific modifications.

## Project Target

This project aims to bridge recent advances in discrete diffusion modeling with biological sequence generation. Specifically, it seeks to:

1. reproduce the original SEDD framework faithfully;
2. extend the method to biological sequence datasets through fine-tuning and domain adaptation;
3. analyze the suitability of discrete diffusion for modeling structured biological token distributions; and
4. identify practical improvements that enhance generation fidelity, optimization stability, and biological relevance.

## Why Discrete Diffusion?

Biological sequences are naturally discrete objects, formed by tokens such as DNA bases, RNA bases, or amino acids. Unlike continuous diffusion, discrete diffusion operates directly in token space, making it a more natural choice for sequence generation tasks.

SEDD is particularly appealing because it provides a principled score-based framework for discrete data and is implemented in a modular way that supports further research. This makes it well suited not only for reproducing the original method, but also for exploring domain-specific improvements on biological sequence datasets. :contentReference[oaicite:2]{index=2}