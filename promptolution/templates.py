EVOPROMPT_DE_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1."""

EVOPROMPT_DE_TEMPLATE_TD = """Please follow the instruction step-by-step to generate a better prompt for the following task: The dataset consists of movie reviews with five levels of sentiment labels: terrible, bad, neutral, okay, good, and great. The task is to classify each movie review into one of these five sentiment categories. The class mentioned first in the response of the LLM will be the prediction.
Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt for the following task: <task_desc>
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1."""

EVOPROMPT_GA_TEMPLATE = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1."""

EVOPROMPT_GA_TEMPLATE_TD = """Please follow the instruction step-by-step to generate a better prompt for the following task: The dataset consists of texts to be simplified. The meaning of the texts is to be kept.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt for the following task: <task_desc>
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1."""

OPRO_TEMPLATE = """Your task is to generate an instruction for the following task:
<task_description>

Below are some previous instructions with their scores. The score ranges from 0 to 100.

<old_instructions>

Here are some examples of the target dataset:
<examples>

Generate a new instruction bracketed with <prompt> and ending it with </prompt> that is different from all the instructions above and has a higher score than all the instructions above. The instruction should be concise, effective, and generally applicable to the task described.

Your new instruction:"""

PROMPT_VARIATION_TEMPLATE = """Generate a single variation of the following instruction while keeping the semantic meaning.
Generate the variation starting with <prompt> and ending with </prompt> tags.

Input: <prev_prompt>

Output:"""

PROMPT_CREATION_TEMPLATE = """You are asked to give the corresponding prompt that gives the following outputs given these inputs.
Return it starting with <prompt> and ending with </prompt> tags.
Include the name of the output classes in the prompt.

<input_output_pairs>

The instruction was"""
