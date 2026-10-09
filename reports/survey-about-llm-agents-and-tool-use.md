# Comprehensive Survey on Large Language Model Agents and Tool Use

## TL;DR
- **Synergistic Execution**: Interleaving reasoning traces with external tool actions (e.g., ReAct) and structured planning (e.g., Plan-and-Solve) overcomes the inherent hallucinations and error propagation of isolated LLMs [1][2].
- **Dynamic & Asynchronous Frontiers**: Modern frameworks extend beyond static API libraries through dynamic code generation (e.g., DynaSaur) and asynchronous multi-task tool calling (e.g., AsyncTool) [3][4].
- **Robust Multi-Agent Collaboration**: Empirical studies demonstrate that genuine multi-agent cooperation requires specialized communication protocols and long-horizon coordination over 50+ rounds rather than naive agent scaling [5][6].
- **Security & Evaluation Auditing**: Vulnerabilities such as tool selection hijacking (ToolHijacker) and evaluation unreliability necessitate runtime least-privilege enforcement (AgenTRIM) and rigorous validation benchmarks (SWE-Bench Pro Verified, TestPrism) [7][8][9][10].

## Background
The paradigm of Large Language Model (LLM) agents represents a fundamental shift from static text generation to autonomous, goal-directed problem solving. While standard LLMs excel at parametric recollection and zero-shot reasoning, they remain bottlenecked by static training distributions, temporal knowledge cutoffs, and an inability to directly interact with external physical or digital environments. To bridge this gap, agentic frameworks couple LLMs with external tool ecosystems—ranging from web search APIs and calculators to software interpreters and graphical user interfaces (GUIs). 

Foundational developments in this domain established that prompting alone is insufficient for multi-step tasks. Seminal architectures such as **ReAct** pioneered the synergy between internal reasoning ("Thought") and external execution ("Action"), allowing models to ground generated hypotheses in verifiable environment feedback [1]. Complementing this, **Plan-and-Solve (PS)** prompting introduced explicit upfront task decomposition, mitigating missing-step errors and premature convergence in complex reasoning workflows [2]. Furthermore, techniques like **Reflexion** and **Agent-R** introduced verbal reinforcement learning and Monte Carlo Tree Search (MCTS) self-training, equipping agents with episodic memory and iterative self-correction mechanisms to recover from execution failures [11][12].

## Foundational Architectures and Reasoning Loops
The core engine of an LLM agent relies on continuous interaction loops that ingest environmental observations, deliberate over state representations, and issue precise tool invocations. The ReAct framework formalized this by alternating between text reasoning and tool actions, demonstrating that reasoning traces help agents formulate more robust action queries while action outputs ground reasoning against hallucination [1]. However, static prompt-based interleaving often degrades over long horizons due to compounding error propagation.

To address error accumulation, recent architectures incorporate structured planning and self-reflection. Plan-and-Solve prompting enforces a two-stage paradigm where the agent first decomposes a macro-goal into ordered sub-tasks before executing them sequentially, significantly reducing calculation and omission errors in arithmetic and symbolic domains [2]. When execution failures occur, **Reflexion** leverages linguistic feedback to generate verbal self-critiques stored in episodic memory, enabling agents to dynamically adjust their strategy across trial iterations without weight updates [11]. Building upon this, **Agent-R** integrates MCTS-guided self-training, allowing smaller open-weight models to systematically explore alternative reasoning branches and internalize self-reflection policies during training [12].

## Dynamic Tool Creation and Asynchronous Function Calling
Traditional agent deployments are bounded by pre-defined API libraries, which restrict flexibility when encountering out-of-distribution problems. To overcome this limitation, **DynaSaur** empowers LLMs with dynamic action creation, enabling models to write, debug, and execute custom code artifacts and procedural programs on the fly [3]. This capability allows agents to synthesize specialized subroutines tailored to novel operational environments rather than relying exclusively on fixed static tool sets.

Concurrently, real-world deployment requires handling concurrent tool requests and delayed responses. The **AsyncTool** benchmark highlights that standard synchronous function calling falls short in multi-task scenarios characterized by latency and asynchronous dependencies, necessitating enhanced temporal reasoning and task coordination mechanisms [4]. To unify these capabilities, **UniToolCall** standardizes tool-use representation, synthetic data generation, and evaluation pipelines, providing a cohesive framework for training models across heterogeneous tool domains [13]. At the inference layer, infrastructure innovations such as **SparseEngine** employ shared lifecycle contracts and sparse attention methods to mitigate KV-cache memory and attention bottlenecks during long-context agent execution [14].

## Multimodal Grounding and Real-World Environment Benchmarks
As agent applications expand from text-based APIs to complex digital workflows, evaluation has shifted from academic QA datasets to immersive, open-ended environments. **OSWorld** established a scalable computer environment for evaluating multimodal agents across operating systems and desktop applications, revealing substantial performance gaps between human users and state-of-the-art vision-language models [15]. Similarly, **AgencyBench** measures agent autonomy across 1M-token real-world contexts using sandbox environments and automated user simulation [16].

In the visual domain, **OneSearch-VL** introduces the Visually Grounded Evidence Graph (VGEG) to unify single-image, multi-image, and video deep research, preserving dependencies across visual anchors and external retrieval [17]. Furthermore, multi-agent embodied environments such as **AgentWorld** demonstrate that long-horizon collaboration across 3 to 20 agents requires specialized coordination protocols and resource sharing over 50+ interaction rounds [5]. However, comparative workflow analyses by **Do More Agents Help?** caution that naively scaling the number of agents yields minimal performance gains without protocol-aligned execution structures [6].

## Fine-Tuning Methodologies and Training Protocols
Supervised fine-tuning (SFT) for agentic workflows has traditionally suffered from fragmented, domain-specific datasets. The **Agent Data Protocol (ADP)** addresses this by establishing a unified formatting standard for heterogeneous training traces, tool invocation logs, and multi-turn interaction histories, resulting in improved cross-domain generalization [18]. 

Concurrently, software engineering benchmarks have undergone rigorous auditing to ensure evaluation integrity. **SWE-Bench Pro Verified** addresses reward hacking, test leakage, and ambiguous problem statements in repository-level code generation, demonstrating that clean evaluation scopes significantly alter model rankings [9]. In the same vein, **TestPrism** introduces the Joint Success Function metric, evaluating coding agents against thousands of candidate implementations rather than a single reference solution [10].

## Security, Vulnerabilities, and Alignment in Tool Use
Empowering LLMs with execution privileges introduces severe security attack surfaces. The **ToolHijacker** framework demonstrates that malicious actors can manipulate tool selection by injecting rogue tool descriptions or unauthorized endpoints into the agent's accessible library, successfully subverting standard prompt-filters [7]. 

To mitigate these risks, **AgenTRIM** implements offline capability reconstruction and runtime least-privilege enforcement, analyzing required tool permissions per task context and blocking unauthorized invocations to restrict the blast radius of compromised agents [8]. These security mechanisms, combined with rigorous evaluation validity audits such as **Benchmarking the Benchmarks**, highlight the critical need for robust verification suites and principled alignment strategies in production-grade agentic systems [19].

## Trends and Open Problems
The landscape of LLM agents and tool use is characterized by rapid paradigm shifts and persistent technical bottlenecks:
1. **From Static APIs to Dynamic Execution**: Moving beyond fixed function registries toward self-synthesizing code artifacts and runtime environment extension [3].
2. **Long-Horizon Multi-Agent Coordination**: Transitioning from superficial multi-agent prompting to protocol-aligned, robust collaborative ecosystems capable of handling 50+ interaction rounds [5][6].
3. **Rigorous Evaluation & Security Hardening**: Eliminating benchmark leakage, evaluator misalignment, and tool selection vulnerabilities through verified task suites and least-privilege runtime sandboxes [19][7][8][9].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-06)
[2] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. arxiv. https://arxiv.org/abs/2305.04091 (2023-05-06)
[3] DynaSaur: Large Language Agents Beyond Predefined Actions. arxiv. https://arxiv.org/abs/2411.01747 (2024-11-04)
[4] AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios. hf-search. https://huggingface.co/papers/2605.27995 (2026-05-27)
[5] AgentWorld: Benchmarking Long-Horizon Collaboration of Multi-agent LLMs. hf-search. https://huggingface.co/papers/2609.31590 (2026-09-25)
[6] Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows. hf-search. https://huggingface.co/papers/2606.05670 (2026-06-04)
[7] Prompt Injection Attack to Tool Selection in LLM Agents (ToolHijacker). hf-search. https://huggingface.co/papers/2504.19793 (2025-08-24)
[8] AgenTRIM: Tool Risk Mitigation for Agentic AI. hf-search. https://huggingface.co/papers/2601.12449 (2026-01-18)
[9] SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents. hf-search. https://huggingface.co/papers/2609.08149 (2026-09-08)
[10] TestPrism: Rethinking Test Evaluation Beyond a Single Reference. hf-daily. https://huggingface.co/papers/2610.12289 (2026-10-08)
[11] Reflexion: Language Agents with Verbal Reinforcement Learning. arxiv. https://arxiv.org/abs/2303.11366 (2023-03-20)
[12] Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training. arxiv. https://arxiv.org/abs/2501.11425 (2025-01-20)
[13] UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents. hf-search. https://huggingface.co/papers/2604.11557 (2026-04-13)
[14] SparseEngine: Sparse-First Inference Engine. hf-daily. https://huggingface.co/papers/2609.39068 (2026-09-30)
[15] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. hf-search. https://huggingface.co/papers/2404.07972 (2024-04-11)
[16] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts. hf-search. https://huggingface.co/papers/2601.11044 (2026-01-16)
[17] OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video. hf-daily. https://huggingface.co/papers/2610.12419 (2026-10-08)
[18] Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents. hf-search. https://huggingface.co/papers/2510.24702 (2025-10-28)
[19] Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation. hf-search. https://huggingface.co/papers/2607.02577 (2026-06-30)
