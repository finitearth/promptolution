# Design Decisions
Here we document all design decision we made during setting up our experiment, which micht deviate from the original paper approaches.

- use a subsample of 20 prompts for each iteration (this saves us a lot of token during execution of the loop, because all samples have to be evaluated in each evolutionary loop) -> cost reduction
- use population size 10 (as we can see from the ablation studies in the papaer that this works quite well) - the initial population is subsampled with the same seed as used for the subsampling of the dev set
- use the first occurence of the class in the response as the classification answer (this is also told to the LLM in the task description)
- we use Llama 3 instead of the more current 3.1 due to faster inference times and lower API costs per million token

#  Experiment Design
We decided to do two sepearate experiments both aiming to answer different research questions:
1.) Can we use a small LLM during optimization (cost efficiency)
2.) Can an additional Task description help the overall performance? (we only test this for small LLMs to save costs)

# Limitations
- we consider the score a prompt gained in the step it was created as its "true" value. We do not reevaluate that prompt, meaning "it could have been lucky" in getting 20 easy down stream examples.
- once a prompt hits 100% in a test metric, it will be chosen as the final prompt. --> no exploration no more (especially in the case of having large llm as evaluator)