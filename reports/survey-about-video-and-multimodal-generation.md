# Comprehensive Survey on Video and Multimodal Generation: Architectures, Open Models, Control, and Evaluation

## TL;DR
- Video and multimodal generation has fundamentally transitioned from 3D U-Nets to Diffusion Transformers (DiTs) and Flow-based architectures, enabling scalable, high-resolution spatiotemporal modeling [1][2][3].
- Spatiotemporal compression relies heavily on advanced 3D VAEs and compatible latent video encoders that balance reconstruction fidelity with token efficiency [1][4][5].
- Open-source and community models (e.g., CogVideoX, LTX-Video) leverage expert transformer blocks and efficient sparse attention mechanisms (such as MC-Sparse) to achieve realtime generation and close the dense-sparse quality gap [1][4][6].
- Long-horizon video synthesis and sustained dynamics require specialized training paradigms like teacher forcing (LongTake) and rigorous human-aligned evaluation suites (VBench-2.0, Video-Bench) [7][8][9].
- Controllable generation has expanded from spatial adapters (ControlNeXt) to modality-agnostic camera control (TriMotion) and joint audio-video benchmarks (VABench, TAVGBench) [10][11][12].

## Background
The rapid advancement of generative artificial intelligence has unlocked unprecedented capabilities in synthesizing high-definition visual content, moving beyond static image generation into temporal video and rich multimodal synthesis. Historically, early video generation models relied on 3D convolutional U-Net architectures conditioned on text embeddings. While effective for short, low-resolution clips, U-Nets faced inherent bottlenecks in scaling parameter capacity and capturing long-range temporal dependencies. 

The paradigm shifted with the adoption of Diffusion Transformers (DiTs) and Flow Matching models, originally popularized in image generation and subsequently scaled to video via architectures like Sora, CogVideoX [1], and Lumina-T2X [3]. These models replace convolutional backbones with transformer blocks operating on compressed latent tokens extracted via 3D Variational Autoencoders (3D VAEs) [1][5]. Concurrently, the rise of open-source initiatives (such as LTX-Video [4]) and specialized conditioning mechanisms (ControlNeXt [12], Control-A-Video [13]) has democratized research into video generation, enabling precise camera control, human-object interactions, and audio-visual synchronization.

## Foundational Architectures and Theoretical Modeling
Modern video generation architectures are grounded in the unification of diffusion processes, flow matching objectives, and transformer-based sequence modeling. Unlike image generation where spatial dimensions dominate, video generation introduces an additional temporal axis that exponentially increases computational complexity.

A cornerstone of modern video DiT models is joint spatiotemporal tokenization. CogVideoX [1] introduced a 3D Variational Autoencoder (3D VAE) that simultaneously compresses video data across spatial and temporal dimensions, feeding compact latent tokens into an expert transformer backbone equipped with 3D full attention. Similarly, LTX-Video [4] utilizes an optimized Video-VAE paired with causal and bidirectional transformer blocks to achieve realtime latent diffusion. To handle arbitrary resolutions, aspect ratios, and durations, Lumina-T2X [3] employs flow-based large diffusion transformers coupled with flexible tokenization schemes. 

Beyond purely generative DiTs, Video Diffusion Transformers (VDT) [2] formulated video generation and zero-shot editing as conditional mask modeling tasks within a diffusion framework, demonstrating superior scalability compared to traditional U-Net baselines. Furthermore, xGen-VideoSyn-1 [14] and CV-VAE [5] highlight the critical importance of spatial-temporal divide-and-merge strategies and compatible latent representations to ensure that high-compression video VAEs do not introduce temporal artifacts or blur motion dynamics.

## Open-Source Models, Efficiency, and Long-Horizon Dynamics
As foundational architectures scaled, the open-source community accelerated empirical research through models addressing efficiency, sparse attention, and long-horizon stability. One of the primary computational bottlenecks in video DiTs is the quadratic complexity of self-attention across massive spatiotemporal token grids. MC-Sparse [6] addresses this by introducing Meta-Cached Sparse Attention, a training-free framework that deconstructs the dense-sparse attention gap and recovers generation fidelity without requiring expensive model retraining.

Maintaining temporal coherence and sustained motion over extended durations (long-horizon video generation) has historically plagued autoregressive video models, which often degenerate into static frames or blur over time. LongTake [7] tackles this limitation via a two-stage training pipeline incorporating Long-Horizon Teacher Forcing (TF) on curated video datasets, ensuring stable scene evolution during extended video creation and interactive simulation. In parallel, specialized domain models such as PAMI [15] utilize Part-Anchored Motion frameworks to coordinate human-object interactions (HOI) explicitly, preventing object drift and penetration artifacts common in implicit trajectory predictions.

## Benchmarks, Evaluation Suites, and Human Alignment
Rigorous evaluation in video generation has evolved from pixel-level metrics (such as FID and FVD) toward multidimensional, human-aligned benchmark suites that measure intrinsic faithfulness, physical plausibility, and semantic alignment. 

VBench-2.0 [8] establishes a comprehensive evaluation framework assessing intrinsic faithfulness across human fidelity, controllability, creativity, and physical commonsense reasoning. Complementing automated metric suites, Video-Bench [9] leverages Multimodal Large Language Models (MLLMs) to systematically evaluate video generation quality against human perceptual preferences, achieving higher correlation with subjective human judgments. 

For specialized video domains, benchmark suites test specific physical and cross-modal dynamics. ChronoMagic-Bench [10] evaluates metamorphic and temporal transformations in text-to-time-lapse video generation, exposing limitations in long-term physical consistency. Furthermore, the emergence of joint audio-visual generation has driven the creation of dedicated testbeds like VABench [12, note: mapped from source 12] and TAVGBench [11], which benchmark cross-modal synchronization, audio fidelity, and semantic relevance across text-to-audio-video tasks.

## Controllable Generation, Camera Control, and Multimodal Conditioning
Controlling the output of video generation models beyond text prompts is vital for practical deployment in filmmaking, simulation, and interactive media. Early spatial control adapters (such as ControlNet) have been adapted and optimized for video through efficient architectures like ControlNeXt [12], which employs cross-normalization to achieve fast training convergence and minimal computational overhead. Control-A-Video [13] similarly extends spatial conditioning into the temporal domain using factorized attention and residual noise initialization.

Recent advancements also target spatial-temporal upscaling and camera trajectory manipulation. VEnhancer [16] provides generative space-time enhancement, simultaneously upscaling spatial resolution and frame rate using unified video diffusion conditioning. For camera control, TriMotion [17] introduces a modality-agnostic embedding space that unifies text, pose, and camera trajectory inputs, enabling flexible, high-fidelity camera maneuvers (panning, zooming, orbiting) without requiring rigid parametric inputs.

## References
[1] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2024-08-12)
[2] VDT: General-purpose Video Diffusion Transformers via Mask Modeling. arxiv. https://arxiv.org/abs/2305.13311 (2023-05-22)
[3] Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers. arxiv. https://arxiv.org/abs/2405.05945 (2024-05-09)
[4] LTX-Video: Realtime Video Latent Diffusion. arxiv. https://arxiv.org/abs/2501.00103 (2024-12-30)
[5] CV-VAE: A Compatible Video VAE for Latent Generative Video Models. arxiv. https://arxiv.org/abs/2405.20279 (2024-05-30)
[6] MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers. hf-daily. https://huggingface.co/papers/2610.06801 (2026-10-05)
[7] LongTake: Learning to Sustain Dynamics in Long-Horizon Video Generation. hf-daily. https://huggingface.co/papers/2609.38562 (2026-09-29)
[8] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. hf-search. https://huggingface.co/papers/2503.21755 (2025-03-27)
[9] Video-Bench: Human-Aligned Video Generation Benchmark. hf-search. https://huggingface.co/papers/2504.04907 (2025-04-07)
[10] VABench: A Comprehensive Benchmark for Audio-Video Generation. hf-search. https://huggingface.co/papers/2512.09299 (2025-12-10)
[11] TAVGBench: Benchmarking Text to Audible-Video Generation. hf-search. https://huggingface.co/papers/2404.14381 (2024-04-22)
[12] ControlNeXt: Powerful and Efficient Control for Image and Video Generation. hf-search. https://huggingface.co/papers/2408.06070 (2024-08-12)
[13] Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. hf-search. https://huggingface.co/papers/2305.13840 (2024-05-23)
[14] xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations. arxiv. https://arxiv.org/abs/2408.12590 (2024-08-22)
[15] PAMI: Part Anchored Motion for Text to Human-Object Interaction Generation. hf-daily. https://huggingface.co/papers/2609.38466 (2026-09-29)
[16] VEnhancer: Generative Space-Time Enhancement for Video Generation. hf-search. https://huggingface.co/papers/2407.07667 (2024-07-10)
[17] TriMotion: Modality-Agnostic Camera Control for Video Generation. hf-search. https://huggingface.co/papers/2606.20774 (2026-06-18)
