# Extending Evaluations of EvoPrompt for Prompt Tuning 

Timo Heiß, Moritz Schlager, Tom Zehle 

## Motivation 
The rapid development of language models has revolutionized data science and natural language processing. However, optimizing prompts for specific tasks remains a challenge. Our project, "Extending Evaluations of EvoPrompt for Prompt Tuning," aims to address this challenge by enhancing the EvoPrompt framework. Our research questions are:
- Can integrating downstream task descriptions in the meta-prompt improve the optimization process and prevent degeneration? 
- How do different metamodels (i.e., different models than the downstream model, as well as varying parameter counts) affect the performance of EvoPrompt in prompt tuning? 

## Previous and Related Work 
Previous research has demonstrated the potential of evolutionary algorithms in prompt tuning. EvoPrompt has shown promising results, yet its application and evaluation have been limited. Studies have highlighted the need for better contextual understanding within the optimization process and the necessity of robust benchmarks to evaluate performance effectively.

EvoPrompt [Guo et Al. 2024] uses evolutionary algorithms for discrete prompt optimization. It leverages the powerful language processing capabilities of LLMs and the efficient optimization performance of evolutionary algorithms, significantly outperforming human-engineered prompts and existing automatic prompt generation methods by up to 25% and 14%, respectively. 

In [Yang et Al. 2024] the performance difference of including the task description into the meta prompt was evaluated. A promising performance boost was shown, however is missing the validation in various datasets. 

## Methodology 
Our methodology will involve:  
- Extending EvoPrompt with downstream task descriptions: 
    - Integrating detailed task descriptions within the meta-prompt to provide better context for optimization 
    - Evaluating the impact on the optimization process and resulting prompt performance by measuring performance on classification benchmarks used in the original paper 
- Analyzing Performance with different Meta-models: 
    - Testing various meta-models within EvoPrompt to identify correlations and performance impacts on beforementioned benchmarks
    - Using a different model than the downstream model 
    - Investigating the feasibility of using smaller meta-models for larger downstream models 
 
## Goals and Objectives 
Our main goals and objectives are: 
- Enhance EvoPrompt: Improve the optimization process by including downstream task descriptions, hypothesizing that this will lead to more meaningful operations and reduced degeneration 
- Expand Benchmarking: Validate the performance of EvoPrompt across multiple benchmarks, with hypotheses on maintaining high performance and improving classification through prompt tuning 
- Meta-model Analysis: Determine the impact of different meta-models on EvoPrompt's performance and explore the potential for using varied metamodel and downstream model combinations 

## Evaluation 
For the evaluation, we will utilize the API for GPT-3.5-Turbo, GPT-4o (https://api.openai.com), Claude 3 Haiku and Claude 3 Opus (https://api.anthropic.com).  

This will allow us to: 
- Assess Performance Across Models: Compare the effectiveness of EvoPrompt when using different sizes of meta-models – keeping downstream model constant to GPT-4o 
- Evaluate Scalability: Determine if the improvements observed with larger models can be replicated with smaller models, thereby assessing the scalability and generalizability of the approach 
- Benchmark Comparison: Use the classification performance on various benchmarks to validate our hypotheses regarding EvoPrompt’s robustness and efficiency in prompt tuning 

## Sources 

[Guo et Al. 2024] Guo, Qingyan; Wang, Rui; Guo, Junliang; Li, Bei; Song, Kaitao; Tan, Xu; Liu, Guoqing; Bian, Jiang; u. a.: Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers, arXiv (2024). — arXiv:2309.08532 [cs] 

[Yang et Al. 2024] - Yang, Chengrun; Wang, Xuezhi; Lu, Yifeng; Liu, Hanxiao; Le, Quoc V.; Zhou, Denny; Chen, Xinyun: Large Language Models as Optimizers, arXiv (2024). — arXiv:2309.03409 [cs] 