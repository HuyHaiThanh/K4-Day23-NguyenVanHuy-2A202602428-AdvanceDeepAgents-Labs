# Survey of World Models: Foundations, Key Advances, and Empirical Scope

## TL;DR
- World models facilitate agent learning and planning via generative prediction of environment dynamics [1][2][3].
- Recent architectural advances highlight JEPA-TTT adaptation for robustness, GreenWAM foundation models for cross-robot tasks, mechanistic approaches to video-state dynamics, and WorldRecon's multiview mesh 3D modeling [4][5][6][7][8].
- Evaluation remains highly application-specific with separate developments in robotics, video, and 3D/4D modeling; universal benchmarks and off-the-shelf maturity are not supported by evidence [2][3][6][7][8].

## Background
World models are internal generative models that learn to simulate environment states from interaction data, often enabling agents to forecast the consequences of actions before real-world execution [1][2][3]. The foundational work by Ha & Schmidhuber (2018) introduced recurrent world models, supporting policy evolution entirely within agents' learned "dream" environments, a paradigm improving sample efficiency without requiring direct access to the original environment [1]. Contemporary surveys further dissect architectures (RNNs, VAEs, transformers), their mathematical roots (latent dynamics, action-conditioned transitions), and their varied applications in robotics and beyond [2][3].

## Theme 1: Mechanistic Modeling for Video State and Dynamics
Mechanistic modeling explores how neural generative models, particularly in video, capture and predict temporal state transitions. This line of work emphasizes the structure and interpretability of state evolution, highlighting challenges such as persistence, control fidelity, and high-dimensional visual prediction [4]. Recent surveys clarify that while video generation models overlap with world modeling goals, their empirical focus is video-specific, prioritizing accuracy of visual and motion dynamics rather than general agent capability [4][8].

## Theme 2: Dynamics Adaptation in JEPA-TTT
Joint-Embedding Predictive Architectures (JEPAs) with persistent Test-Time Training (JEPA-TTT) represent an advance in model robustness under rapidly changing environment dynamics [5]. By adapting world models online as the deployment context shifts, JEPA-TTT seeks to maintain policy fidelity and planning accuracy. The work is empirically grounded in video-based and low-level control environments—generalization beyond these remains open [5].

## Theme 3: Foundation Models for Cross-Robot World and Action (GreenWAM)
GreenWAM describes a foundation world-action model trained across diverse robot families, supporting manipulation and cross-domain transfer in robotics [6]. The core achievement is unifying perception and action generation with a single model evaluated on a range of robot morphologies and environments, tested within robotics manipulation [6]. Evidence for foundation model effectiveness is specific to GreenWAM's robotic evaluation protocols and does not claim universal industrial deployment or broad multimodal supremacy.

## Theme 4: 3D/4D World Modeling and Multiview Mesh Reconstruction
WorldRecon addresses generative mesh modeling using multiview data, reconstructing 3D scenes for large-scale world understanding [7]. This approach is evaluated on mesh quality, scale, and accuracy from multiview sources. Related work in 3D/4D world model surveys confirms vibrant progress in reconstruction pipelines and benchmarks, while recognizing the application scope is still specialized and domain-specific [8].

## Trends and open problems
- Ha & Schmidhuber (2018) showcase policy learning from internal dream environments, which subsequent surveys identify as a central concept for sample-efficient learning [1][2][3].
- Video state/dynamics modeling [4] and mesh-based 3D/4D world reconstruction [7][8] are parallel lines, each empirically bound to their domains.
- JEPA-TTT adaptation [5] and GreenWAM [6] illustrate advances in dynamics and cross-robot generalization, with effectiveness reported within their respective benchmarks.
- No current evidence—across these cited sources—demonstrates universal cross-domain standardization or industrial-scale, broad deployment of world models; open challenges remain in evaluation, real-world transfer, and interpretability [2][3][6][8].

## References
[1] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[2] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[3] World Model for Robot Learning: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2605.00080 (2026-04-30)
[4] A Mechanistic View on Video Generation as World Models: State and Dynamics. hf-search. https://huggingface.co/papers/2601.17067 (2026-01-22)
[5] JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts. hf-search. https://huggingface.co/papers/2610.00722 (2026-09-30)
[6] GreenWAM: A World Foundation Model for Robotics. hf-daily. https://huggingface.co/papers/2610.18774 (2026-10-13)
[7] WorldRecon: Generative Multi-View Mesh Reconstruction for Large-Scale 3D World. hf-daily. https://huggingface.co/papers/2610.18846 (2026-10-13)
[8] 3D and 4D World Modeling: A Survey. web. https://worldbench.github.io/assets_common/papers/survey.pdf (2023)
