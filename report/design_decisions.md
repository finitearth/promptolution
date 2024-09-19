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

GPT-Experiment (using GPT-4o as downstream model for evaluating the best prompts):
1.) we do this only on 4 of the initial 7 experiments (agnews, cr, sst-5, trec)  
    - mr & sst2 were removed, because it is a binary sentiment classification like cr
    - subj removed because it is also a binary problem and despite no sentiment analysis quite similar
    - trec was not removed, but some of the best prompts are the entire template due to parsing issues
2.) only use the results from the evopromptde optimizer, since its performance was better in previous experiments
3.) we only perform this for the best prompts across all 3 seeds per dataset resulting in only 4 best prompts to finally evaluate
4.) we only look at the results from utilizing the Llama 70B model

This results in a total of:

200 samples * 1 downstream_llm * 1 optimizer * 4 datasets * 3 seeds = 2400 iterations
we assume an average token size of 35 tokens per sample + 25 per prompt = 60 tokens
--> 144000 tokens


# Limitations
- we consider the score a prompt gained in the step it was created as its "true" value. We do not reevaluate that prompt, meaning "it could have been lucky" in getting 20 easy down stream examples.
- once a prompt hits 100% in a test metric, it will be chosen as the final prompt. --> no exploration no more (especially in the case of having large llm as evaluator)
- limit of token length lead to unfinished prompts -> <prompt> was never reached, therefore whole metaprompt is prompt

