# A Rigorous Survey of LLM Agents and Tool Use

## TL;DR
- Foundational definitions converge: LLM agents are autonomous, goal-directed systems built on large language models with tool use as a central action modality [1][2].
- Recent years (2024–2026) have seen advances in dynamic, adaptive, and robust tool use, including proactive tool retrieval, error recovery via feedback, tool synthesis, and reflection [3][4][5][6][7].
- Core system components focus on agent-module synthesis—namely API/action interfaces, planning/policy modules, memory, and grounding strategies for tool-enabled action—and demonstrate spatial grounding capabilities for complex 3D environments [1][2][8].
- Key open challenges: generalizable reliability over long horizons, scalable architectures for evolving toolsets, robust multi-agent coordination (empirical evidence from issue tracking shows orchestration/execution as the most common challenge and workflow/tool integration/memory as frequent causes [9]), safety and governance [10], process-centric evaluation [9][10], and planning over tool hypergraphs [11].

## Background

Large language model (LLM) agents are systems built on modern language models, such as GPT or open-source alternatives, designed to act autonomously in their environment by integrating perception, reasoning, planning, and action [1][2]. Tool use empowers these agents to interact not only via text but through structured APIs, plugins, external software, and other systems. This transforms a language model from a passive predictor into an adaptive, goal-directed agent. The notion originates in classic AI agent frameworks, with contemporary work expanding into multi-step tool orchestration, open-world environments, and dynamic tool creation [1][2]. Early LLM agents, such as Auto-GPT and BabyAGI, exemplified the centrality of tool use in extending LLM capabilities beyond language modeling to genuine environment interaction [2].

## Foundational Concepts of LLM Agents and Tool Use

Foundational sources agree that an LLM agent is an LLM augmented with modules and interfaces for goal pursuit, environment perception and tracking (potentially across modalities), and crucially, action—including via tool use [1]. Definitions adapt classical AI perspectives: agents sense the world, make decisions, and act, with tool use as the principal mechanism for meaningful change [2]. Action modules may include single or multi-tool invocation, workflow execution, error recovery, and in advanced forms, dynamic tool creation [2].

General agent-module synthesis includes identity/profile management, episodic memory, strategic planning, and action interfaces. Tool use features centrally in the action module, with fundamental surveys emphasizing iterative prompting, feedback learning, and continuous skill acquisition [1][2]. These works mark progression from basic tool invocation to multi-tool orchestration or workflow execution as key maturity steps. As competence advances, agents may move from simple supervised fine-tuning of action calls to dynamic tool composition and error recovery [1][2].

Boundaries around what constitutes tool use are evolving, as agents begin creating tools, deciding when to use or synthesize new ones, and dynamically combining action modules [2]. These developments remain firmly connected to AI agent theory but expanded uniquely by the practical demands of LLM-based tool-enabled agents [1][2].

## Core Architectures and Methods for Tool Use

Surveyed technical approaches coalesce on agent-module synthesis: integrating APIs/action interfaces, planning and policy components, memory/episodic tracking, and error-recovery/reflection modules [1][2]. Architectures are designed to link LLMs with flexible tool invocation and the ability to select and parameterize actions according to context.

**API and Function Integration**: Early and representative architectures emphasize accurate and generalizable API tool invocation. Synthetic data for tool-oriented tasks and benchmark environments are common training tools [1][2].

**Planning and Module Interaction**: Various agent proposals highlight the role of planning modules, capable of multi-step forecasting and adaptation based on episodic memory and external feedback [1][2].

**Spatial Grounding Application**: Urbex demonstrates the extension of agentic tool use to spatial grounding—enabling robust, modular interaction in 3D environments. These agents ground multi-step action and perception in structured, spatial scenes, broadening the architecture landscape [8].

Despite these advances, there remain significant gaps: generalization, open-ended orchestration, and proprietary components for routing/validation are recurrent limitations [1][2].

## Recent Advances: Dynamic, Robust, and Adaptive Tool Use (2024–2026)

The 2024–2026 period has seen a focus on robustness, adaptability, and the operation of agents in dynamic, error-prone, and open-ended environments [3][4][5][6][7][8]. Recent contributions include:

**Proactive Error Recovery and Reflection**: Systems like Agent-Reflex introduce proactive reflection, enabling agents to handle realism-induced noise, user interventions, and novel tool feedback [3]. Reflection skills learned in training help improve error recovery and adaptability [3].

**Robust Error Handling via Feedback**: FISSION-GRPO demonstrates RL techniques for converting tool execution errors into direct training feedback, enabling graceful recovery from failures and reducing error loops [4].

**Dynamic Tool Synthesis and Planning**: CAR implements on-the-fly tool creation and strategy updating, allowing agents to adapt to unrehearsed or incomplete environments [5].

**Open-World Tool Use**: ToolOmni (source6) implements proactive tool retrieval for dynamic discovery and use of new tools, showing robust generalization to unseen APIs and adaptation to API changes [6].

**Meta-Learned Adaptation**: ExpG (source7) applies a meta-learning approach, filtering tool-use experiences to inform and adapt future tool selection, with smaller models surpassing larger but static baselines [7].

**Complex Modalities**: Urbex expands LLM agent use to spatial reasoning and tool coordination in large-scale 3D environments, demonstrating the applicability of agentic tool use beyond textual tasks [8].

While these works mark genuine progress, limitations remain: schema- or constraint-dependent tool use is still common, with few systems fully tackling user-defined, open-ended tool environments, and systematic multi-agent evaluation is sparse [5][6][7].

## Trends and open problems

A review of surveys and recent work outlines convergent open research challenges [10][11][9]:

- **Long-Horizon Brittle Planning**: Tool use across multiple steps and evolving goals remains brittle, with increased failure and error compounding for complex tasks [10][11].
- **Generalization and Robustness**: Open challenges persist in adapting to novel/changing tools, error recovery, and robust execution with minimal supervision [4][6][11].
- **Safety, Security, and Governance**: Reviews note ongoing risks, including hallucinated tool calls, privacy, security shortfalls, and insufficient oversight in real-world actions [10].
- **Empirical Issue Tracking**: An empirical issue study of LLM-based multi-agent systems found orchestration and execution to be the most common issue categories; workflow/tool integration and memory were frequent causes, while workflow optimization was the predominant solution strategy [9].
- **Evaluation Blind Spots**: Existing evaluation trends focus mainly on outcome/task success, lacking process-level, safety, latency/cost, or operational comparison [9][10].
- **Credit Assignment and Memory**: Understanding causality in agent actions—especially across long tool chains—and enabling context-aware memory remain major hurdles [11].
- **Open, Adaptive Architectures**: Full open-world, unconstrained tool use, including autonomous discovery, composition, and safe reuse, is not yet realized; dynamic skill adaptation remains an open research frontier [5][6][7].

New benchmarks, adversarial evaluations, and modality expansions are evolving. Research priorities are moving toward scalable, safe, and trustworthy architectures for robust agentic tool use, especially for collaboration and high-stakes application contexts [10][11].

## References
[1] A Survey on Large Language Model based Autonomous Agents. arxiv. https://arxiv.org/abs/2308.11432 (2025-03-02)
[2] The Rise and Potential of Large Language Model Based Agents: A Survey. arxiv. https://arxiv.org/abs/2309.07864 (2023-09-19)
[3] Agent-Reflex: Teaching Language Agents to Act Reflectively. hf-search. https://huggingface.co/papers/2610.17953 (2026-10-13)
[4] Robust Tool Use via FISSION-GRPO: Learning to Recover from Execution Errors. web. https://aclanthology.org/2026.acl-long.1880.pdf (2026-07)
[5] CAR: Empowering Agents with Dynamic Tool Synthesis and Global Trajectory Rectification. web. https://aclanthology.org/2026.findings-acl.869.pdf (2026-07)
[6] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution. web. https://aclanthology.org/2026.acl-long.1736.pdf (2026-07)
[7] Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance. arxiv. https://arxiv.org/abs/2608.03403 (2026-08-04)
[8] Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes. hf-search. https://huggingface.co/papers/2610.18356 (2026-10-13)
[9] Understanding Issues, Causes and Solutions in Open-Source LLM-based Multi-Agent Systems. arxiv. https://arxiv.org/abs/2610.00905 (2026-10-01)
[10] From Language Models to Agentic AI: A Survey of Autonomous, Action-Enabled, and Collaborative LLM Agents. web. https://link.springer.com/article/10.1007/s12559-026-10619-1 (2026-08-24)
[11] HyperAgent: Planning and Acting over Tool-Schema Hypergraphs for Tool-Use LLM Agents. hf-search. https://huggingface.co/papers/2608.02650 (2026-07-31)
