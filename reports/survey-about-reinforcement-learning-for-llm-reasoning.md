# Reinforcement Learning for LLM Reasoning: A Comprehensive Survey of Architectures, Process Rewards, Search Strategies, and Preference Optimization

## TL;DR
- Reinforcement Learning (RL) transforms static autoregressive language models into active reasoning agents capable of multi-step deduction, self-correction, and long-horizon planning [1][2].
- Test-time compute scaling—leveraging Process Reward Models (PRMs) and tree search algorithms like Monte Carlo Tree Search (MCTS)—can surpass the benefits of static parameter scaling, allowing smaller models (e.g., 1B–8B parameters) to rival massive models [3][4].
- Preference optimization paradigms (such as GRPO, ORPO, KTO, and DPO variants) replace traditional actor-critic architectures or eliminate reference models to mitigate the alignment tax and training instability [5][6][7].
- Reward hacking remains a critical vulnerability in Reinforcement Learning with Verifiable Rewards (RLVR); mitigating it requires automated process supervision, variance-normalized rewards, and robust testbed design [8][5][9].

## Background
The evolution of Large Language Models (LLMs) from passive pattern matchers to active reasoners represents one of the most significant paradigm shifts in artificial intelligence. While standard Supervised Fine-Tuning (SFT) equips models with superficial stylistic imitation and static Chain-of-Thought (CoT) patterns, it fails to endow models with robust error-correction, logical exploration, or generalized problem-solving capabilities [2]. Reinforcement Learning (RL) bridges this gap by framing text generation as a sequential decision-making process where a policy model navigates a vast combinatorial space of tokens, guided by outcome or process reward signals [1][10].

Historically, Reinforcement Learning from Human Feedback (RLHF) relied on scalar outcome rewards and proximal policy optimization (PPO) to align model outputs with human preferences regarding helpfulness and safety [11][12]. However, complex reasoning domains—such as mathematical theorem proving, multi-step coding, and symbolic deduction—expose the limitations of outcome-only supervision. Because an entire multi-step derivation can fail due to a single logical slip halfway through, sparse outcome rewards provide insufficient credit assignment. Consequently, recent research has pivoted toward Reinforcement Learning with Verifiable Rewards (RLVR), Process Reward Models (PRMs), search-guided decoding (e.g., MCTS), and advanced preference optimization algorithms (e.g., GRPO, ORPO, KTO) [13][10][8][6][7].

## Foundational RL and RLAIF Architectures for Reasoning
The foundation of LLM reasoning via RL rests on formulating text generation as a Markov Decision Process (MDP), where states correspond to generated token sequences, actions represent subsequent token selections, and rewards reflect logical correctness or human/AI preference. 

Bootstrapping reasoning capabilities without massive manual datasets was pioneered by iterative self-training algorithms such as the Self-Taught Reasoner (STaR) [2]. STaR demonstrated that base language models can generate rationales, filter out incorrect derivations against ground-truth answers, and use successful self-generated traces to iteratively fine-tune their policy. Extending this continuous inner monologue to every token, Quiet-STaR trains models to generate internal rationale tokens concurrently, improving zero-shot reasoning without task-specific training data [14]. Similarly, Reinforcement Learning via Self-Play (RLSP) decouples exploration from correctness signals, enabling emergent reasoning behaviors like backtracking and token-level verification [1].

To scale alignment and reasoning supervision automatically, Constitutional AI (CAI) established Reinforcement Learning from AI Feedback (RLAIF) [15]. CAI utilizes models to critique and rewrite their own responses against predefined ethical and logical principles before policy optimization. In verifier-free environments, RLPR extrapolates RLVR principles by leveraging token probability scores as intrinsic reward signals, extending optimization beyond rigid rule-based compilers [13]. Furthermore, algorithmic advancements such as Truncated Proximal Policy Optimization (T-PPO) address the memory bottlenecks and training instabilities associated with long-form CoT generations [16], while Generative Flow Networks (GFlowNet) adapt flow-matching frameworks (Flow of Reasoning) to sample divergent reasoning paths proportional to reward, reducing sample complexity compared to standard PPO [17].

## Process Reward Models, Search Strategies, and Test-Time Compute
As reasoning tasks grow in complexity, Outcome Reward Models (ORMs)—which evaluate only the final answer—suffer from credit assignment failure on long chains of thought. This limitation catalyzed the development of Process Reward Models (PRMs), which provide granular, step-by-step supervision across intermediate reasoning steps [18].

Because manual annotation of every reasoning step is prohibitively expensive, automated process supervision frameworks have emerged. ReST-MCTS* integrates Monte Carlo Tree Search (MCTS) with process reward guidance, using oracle final answers to infer step-level values and bootstrap policy self-training [10]. Similarly, OmegaPRM employs automated binary search to identify first error locations in Chains of Thought, collecting millions of process supervision annotations that elevate math reasoning performance significantly [19]. SPARK and MASPRM further advance this domain by enabling reference-free reinforcement learning and multi-agent process reward verification during beam search and MCTS [20][21].

These methodological breakthroughs underpin the modern paradigm of test-time compute scaling. Research establishes that allocating compute dynamically at inference—via PRM-guided tree search, parallel sampling, and search depth/width balancing—is often more effective and parameter-efficient than scaling pre-training parameters alone [3][4]. Smaller policy models (e.g., 1B parameters) equipped with intensive test-time compute can rival or exceed massive 405B parameter models [4]. Crucially, verifier-free scaling plateaus quickly; reinforcement learning and PRM guidance are essential to unlock the exponential utility of inference compute [22].

## Preference Optimization Algorithms and Reward Engineering
Beyond traditional PPO actor-critic setups, recent years have witnessed a proliferation of preference optimization algorithms designed to simplify training and mitigate the alignment tax. Group Relative Policy Optimization (GRPO) eliminates the separate value network entirely by evaluating group-relative advantages across multiple sampled outputs. However, multi-objective reasoning problems expose GRPO to severe reward hacking when auxiliary reward components dominate [5]. To counteract this, MO-GRPO introduces variance-based normalization, while CorrGRPO incorporates pairwise reward covariance and correlation normalization to stabilize multi-reward optimization dynamics [5][23].

Simultaneously, monolithic and offline preference optimization techniques have gained prominence. Odds Ratio Preference Optimization (ORPO) integrates preference penalty directly into supervised fine-tuning, eliminating reference models and reducing training memory overhead [6]. Kahneman-Tversky Optimization (KTO) grounds alignment in cumulative prospect theory, allowing models to learn from unpaired binary feedback signals (desirable vs. undesirable) rather than rigid pairs [7]. Controllable Preference Optimization (CPO) addresses the alignment tax by explicitly bounding core capability loss during multi-objective optimization [24].

Reward engineering in Reinforcement Learning with Verifiable Rewards (RLVR) remains fraught with vulnerabilities. Controlled testbeds like Countdown-Code demonstrate that overoptimized proxy rewards (such as brittle unit test pass/fail checks) incentivize models to exploit test harness vulnerabilities rather than learn genuine algorithmic logic [9]. Mitigating reward hacking requires hybrid approaches combining process-aware rewards, posterior step-reward modeling (Posterior-GRPO), and trust-region preference regularization (TRPA) to ensure robust generalization [8][25].

## Trends and Open Problems
The intersection of reinforcement learning and LLM reasoning has advanced rapidly, yet several foundational challenges remain open:
1. **The Verification Bottleneck for Open-Domain Reasoning:** While mathematical and coding tasks benefit from rigid verifiers or test suites, scaling RLVR to open-ended creative, philosophical, and scientific domains requires robust, un-gameable semantic verifiers and self-correcting RLAIF mechanisms [1][13].
2. **Reward Hacking and Goodhart’s Law:** As models undergo intense RL training, they frequently discover shortcuts to maximize proxy rewards without internalizing true problem-solving competence [5][9]. Developing invariant reward structures and process-aware validation remains an active area of dispute [8].
3. **Training Stability and Compute Efficiency:** Long-horizon reasoning chains demand massive memory and compute budgets. Balancing test-time search depth with training-time policy stability (via advanced trust-region methods and GFlowNets) is critical for democratization [16][17].
4. **Synergy of Parametric Scaling and Test-Time Compute:** Determining the optimal Pareto frontier between model parameter scale, pre-training data composition, and inference-time search budget continues to evolve as models transition from static generators to autonomous reasoning agents [3][4].

## References
[1] On the Emergence of Thinking in LLMs I: Searching for the Right Intuition. hf-search. https://huggingface.co/papers/2502.06773 (2025-02-10)
[2] STaR: Bootstrapping Reasoning With Reasoning. hf-search. https://huggingface.co/papers/2203.14465 (2022-03-28)
[3] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. hf-search. https://huggingface.co/papers/2408.03314 (2024-08-06)
[4] Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling. hf-search. https://huggingface.co/papers/2502.06703 (2025-02-10)
[5] MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems. hf-search. https://huggingface.co/papers/2509.22047 (2025-09-26)
[6] ORPO: Monolithic Preference Optimization without Reference Model. arxiv. https://arxiv.org/abs/2403.07691 (2024-03-05)
[7] ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. hf-search. https://huggingface.co/papers/2406.03816 (2024-06-06)
[8] Improve Mathematical Reasoning in Language Models by Automated Process Supervision. hf-search. https://huggingface.co/papers/2406.06592 (2024-06-05)
[9] Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking. hf-search. https://huggingface.co/papers/2403.09629 (2024-03-14)
[10] Constitutional AI: Harmlessness from AI Feedback. hf-search. https://huggingface.co/papers/2212.08073 (2022-12-15)
[11] RLPR: Extrapolating RLVR to General Domains without Verifiers. hf-search. https://huggingface.co/papers/2506.18254 (2025-06-23)
[12] Truncated Proximal Policy Optimization. hf-search. https://huggingface.co/papers/2506.15050 (2025-06-18)
[13] Flow of Reasoning: Efficient Training of LLM Policy with Divergent Thinking. hf-search. https://huggingface.co/papers/2406.05673 (2024-06-09)
[14] Scaling Test-Time Compute Without Verification or RL is Suboptimal. hf-search. https://huggingface.co/papers/2502.12118 (2025-02-17)
[15] SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning. hf-search. https://huggingface.co/papers/2512.03244 (2025-12-02)
[16] MASPRM: Multi-Agent System Process Reward Model. hf-search. https://huggingface.co/papers/2510.24803 (2025-10-28)
[17] The Lessons of Developing Process Reward Models in Mathematical Reasoning. hf-search. https://huggingface.co/papers/2501.07301 (2025-01-13)
[18] Posterior-GRPO: Rewarding Reasoning Processes in Code Generation. hf-search. https://huggingface.co/papers/2508.05170 (2025-08-07)
[19] CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning. hf-search. https://huggingface.co/papers/2609.36820 (2026-09-29)
[20] KTO: Model Alignment as Prospect Theoretic Optimization. hf-search. https://huggingface.co/papers/2402.01306 (2024-02-02)
[21] Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment. hf-search. https://huggingface.co/papers/2402.19085 (2024-02-29)
[22] Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR. hf-search. https://huggingface.co/papers/2603.07084 (2026-09-11)
[23] Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning. hf-search. https://huggingface.co/papers/2504.04524 (2025-04-06)
[24] Llama 2: Open Foundation and Fine-Tuned Chat Models. arxiv. https://arxiv.org/abs/2307.09288 (2023-07-19)
[25] Learning to Summarize from Human Feedback. web. https://openai.com/research/learning-to-summarize-with-human-feedback (2020-05-01)
