# Getting started

## Before you start

In this notebook we give you a short introduction into the workings of promptolution.

We will use the OpenAI-API to demonstrate the functionality of promptolution, however we also provide a local LLM, as well as a vLLM backend. You can also change the `base_url` in the config, in order to use any other api, that follows the OpenAI API standard.

Thanks for giving it a try!

## Installs


```python
# ! pip install promptolution
```

## Imports


```python
import pandas as pd
from promptolution import ExperimentConfig, run_experiment
import nest_asyncio
nest_asyncio.apply() # we need this only because we are in a notebook
```

## set up llms, predictor, tasks and optimizer

Here we set up our dataset. We use the subjectivity dataset from hugging face, but of course here you may want to use your own dataset.

Just make sure, to name the input column "x" and the target column "y", as well as providing a short dataset description.


```python
df = pd.read_csv("hf://datasets/tasksource/subjectivity/train.csv")
df = df.rename(columns={"Sentence": "x", "Label": "y"})
df = df.replace({"OBJ": "objective", "SUBJ": "subjective"})

task_description = "The dataset contains sentences labeled as either subjective or objective. "\
        "The task is to classify each sentence as either subjective or objective. " \
        "The class mentioned first in the response of the LLM will be the prediction."
```

We definied some initial prompts, however you may also take a look at `create_prompts_from_samples` in order to automatically generate them.


```python
init_prompts = [
    'Classify the given text as either an objective or subjective statement based on the tone and language used: e.g. the tone and language used should indicate whether the statement is a neutral, factual summary (objective) or an expression of opinion or emotional tone (subjective). Include the output classes "objective" or "subjective" in the prompt.',
    'What kind of statement is the following text: [Insert text here]? Is it <objective_statement> or <subjective_statement>?',
    'Identify whether a sentence is objective or subjective by analyzing the tone, language, and underlying perspective. Consider the emotion, opinion, and bias present in the sentence. Are the authors presenting objective facts or expressing a personal point of view? The output will be either "objective" (output class: objective) or "subjective" (output class: subjective).',
    'Classify the following sentences as either objective or subjective, indicating the name of the output classes: [input sentence]. Output classes: objective, subjective',
    '_query a text about legal or corporate-related issues, and predict whether the tone is objective or subjective, outputting the corresponding class "objective" for non-subjective language or "subjective" for subjective language_',
    'Classify a statement as either "subjective" or "objective" based on whether it reflects a personal opinion or a verifiable fact. The output classes to include are "objective" and "subjective".',
    'Classify the text as objective or subjective based on its tone and language.',
    'Classify the text as objective or subjective based on the presence of opinions or facts. Output classes: objective, subjective.',
    'Classify the given text as objective or subjective based on its tone, focusing on its intention, purpose, and level of personal opinion or emotional appeal, with outputs including classes such as objective or subjective.',
    "Categorize the text as either objective or subjective, considering whether it presents neutral information or expresses a personal opinion/bias.\n\nObjective: The text has a neutral tone and presents factual information about the actions of Democrats in Congress and the union's negotiations.\n\nSubjective: The text has a evaluative tone and expresses a positive/negative opinion/evaluation about the past performance of the country.",
    'Given a sentence, classify it as either "objective" or "subjective" based on its tone and language, considering the presence of third-person pronouns, neutral language, and opinions. Classify the output as "objective" if the tone is neutral and detached, focusing on facts and data, or as "subjective" if the tone is evaluative, emotive, or biased.',
    'Identify whether the given sentence is subjective or objective, then correspondingly output "objective" or "subjective" in the form of "<output class>, (e.g. "objective"), without quotes. Please note that the subjective orientation typically describes a sentence where the writer expresses their own opinion or attitude, whereas an objective sentence presents facts or information without personal involvement or bias. <output classes: subjective, objective>'
]
```

We will be now using the gpt


```python
token = open("../deepinfratoken.txt", "r").read()
```


```python
config = ExperimentConfig(
    task_description=task_description,
    prompts=init_prompts,
    n_steps=3,
    optimizer="evopromptga",
    api_url="https://api.openai.com/v1",
    llm="gpt-4o-mini-2024-07-18",
    token=token,
)
```


```python
prompts = run_experiment(df, config)
```


    ---------------------------------------------------------------------------

    RateLimitError                            Traceback (most recent call last)

    Cell In[48], line 1
    ----> 1 prompts = run_experiment(df, config)


    File ~\Documents\programming\promptolution\promptolution\helpers.py:32, in run_experiment(df, config)
         30 train_df = df.sample(frac=0.8, random_state=42)
         31 test_df = df.drop(train_df.index)
    ---> 32 prompts = run_optimization(train_df, config)
         33 df_prompt_scores = run_evaluation(test_df, config, prompts)
         35 return df_prompt_scores


    File ~\Documents\programming\promptolution\promptolution\helpers.py:59, in run_optimization(df, config)
         51 task = get_task(df, config)
         52 optimizer = get_optimizer(
         53     predictor=predictor,
         54     meta_llm=llm,
         55     task=task,
         56     config=config,
         57 )
    ---> 59 prompts = optimizer.optimize(n_steps=config.n_steps)
         61 if config.prepend_exemplars:
         62     selector = get_exemplar_selector(config.exemplar_selector, task, predictor)


    File <string>:15, in optimize(self, n_steps)


    File ~\Documents\programming\promptolution\promptolution\optimizers\evoprompt_ga.py:69, in EvoPromptGA._pre_optimization_loop(self)
         67     logger.warning(f"Initial sequences: {seq}")
         68 else:
    ---> 69     self.scores = self.task.evaluate(
         70         self.prompts, self.predictor, subsample=True, n_samples=self.n_eval_samples
         71     ).tolist()
         72 # sort prompts by score
         73 self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]


    File ~\Documents\programming\promptolution\promptolution\tasks\classification_tasks.py:101, in ClassificationTask.evaluate(self, prompts, predictor, system_prompts, n_samples, subsample, return_seq)
         98 ys_subsample = self.ys[indices]
        100 # Make predictions on the subsample
    --> 101 preds = predictor.predict(prompts, xs_subsample, system_prompts=system_prompts, return_seq=return_seq)
        103 if return_seq:
        104     preds, seqs = preds


    File ~\Documents\programming\promptolution\promptolution\predictors\base_predictor.py:57, in BasePredictor.predict(self, prompts, xs, system_prompts, return_seq)
         54 if isinstance(prompts, str):
         55     prompts = [prompts]
    ---> 57 outputs = self.llm.get_response(
         58     [prompt + "\n" + x for prompt in prompts for x in xs], system_prompts=system_prompts
         59 )
         60 preds = self._extract_preds(outputs)
         62 shape = (len(prompts), len(xs))


    File ~\Documents\programming\promptolution\promptolution\llms\base_llm.py:97, in BaseLLM.get_response(self, prompts, system_prompts)
         95 if isinstance(system_prompts, str):
         96     system_prompts = [system_prompts] * len(prompts)
    ---> 97 responses = self._get_response(prompts, system_prompts)
         98 self.update_token_count(prompts + system_prompts, responses)
        100 return responses


    File ~\Documents\programming\promptolution\promptolution\llms\api_llm.py:82, in APILLM._get_response(self, prompts, system_prompts)
         79 def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
         80     # Setup for async execution in sync context
         81     loop = asyncio.get_event_loop()
    ---> 82     responses = loop.run_until_complete(self._get_response_async(prompts, system_prompts))
         83     return responses


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\nest_asyncio.py:98, in _patch_loop.<locals>.run_until_complete(self, future)
         95 if not f.done():
         96     raise RuntimeError(
         97         'Event loop stopped before Future completed.')
    ---> 98 return f.result()


    File ~\AppData\Local\Programs\Python\Python312\Lib\asyncio\futures.py:203, in Future.result(self)
        201 self.__log_traceback = False
        202 if self._exception is not None:
    --> 203     raise self._exception.with_traceback(self._exception_tb)
        204 return self._result


    File ~\AppData\Local\Programs\Python\Python312\Lib\asyncio\tasks.py:316, in Task.__step_run_and_handle_result(***failed resolving arguments***)
        314         result = coro.send(None)
        315     else:
    --> 316         result = coro.throw(exc)
        317 except StopIteration as exc:
        318     if self._must_cancel:
        319         # Task is cancelled right before coro stops.


    File ~\Documents\programming\promptolution\promptolution\llms\api_llm.py:90, in APILLM._get_response_async(self, prompts, system_prompts)
         85 async def _get_response_async(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
         86     tasks = [
         87         _invoke_model(prompt, system_prompt, self.max_tokens, self.llm, self.client, self.semaphore)
         88         for prompt, system_prompt in zip(prompts, system_prompts)
         89     ]
    ---> 90     responses = await asyncio.gather(*tasks)
         91     return [response.choices[0].message.content for response in responses]


    File ~\AppData\Local\Programs\Python\Python312\Lib\asyncio\tasks.py:385, in Task.__wakeup(self, future)
        383 def __wakeup(self, future):
        384     try:
    --> 385         future.result()
        386     except BaseException as exc:
        387         # This may also be a cancellation.
        388         self.__step(exc)


    File ~\AppData\Local\Programs\Python\Python312\Lib\asyncio\tasks.py:314, in Task.__step_run_and_handle_result(***failed resolving arguments***)
        310 try:
        311     if exc is None:
        312         # We use the `send` method directly, because coroutines
        313         # don't have `__iter__` and `__next__` methods.
    --> 314         result = coro.send(None)
        315     else:
        316         result = coro.throw(exc)


    File ~\Documents\programming\promptolution\promptolution\llms\api_llm.py:25, in _invoke_model(prompt, system_prompt, max_tokens, model_id, client, semaphore)
         23 async with semaphore:
         24     messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    ---> 25     response = await client.chat.completions.create(
         26         model=model_id,
         27         messages=messages,
         28         max_tokens=max_tokens,
         29     )
         30     return response


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\resources\chat\completions\completions.py:2032, in AsyncCompletions.create(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, reasoning_effort, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, web_search_options, extra_headers, extra_query, extra_body, timeout)
       1989 @required_args(["messages", "model"], ["messages", "model", "stream"])
       1990 async def create(
       1991     self,
       (...)   2029     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
       2030 ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
       2031     validate_response_format(response_format)
    -> 2032     return await self._post(
       2033         "/chat/completions",
       2034         body=await async_maybe_transform(
       2035             {
       2036                 "messages": messages,
       2037                 "model": model,
       2038                 "audio": audio,
       2039                 "frequency_penalty": frequency_penalty,
       2040                 "function_call": function_call,
       2041                 "functions": functions,
       2042                 "logit_bias": logit_bias,
       2043                 "logprobs": logprobs,
       2044                 "max_completion_tokens": max_completion_tokens,
       2045                 "max_tokens": max_tokens,
       2046                 "metadata": metadata,
       2047                 "modalities": modalities,
       2048                 "n": n,
       2049                 "parallel_tool_calls": parallel_tool_calls,
       2050                 "prediction": prediction,
       2051                 "presence_penalty": presence_penalty,
       2052                 "reasoning_effort": reasoning_effort,
       2053                 "response_format": response_format,
       2054                 "seed": seed,
       2055                 "service_tier": service_tier,
       2056                 "stop": stop,
       2057                 "store": store,
       2058                 "stream": stream,
       2059                 "stream_options": stream_options,
       2060                 "temperature": temperature,
       2061                 "tool_choice": tool_choice,
       2062                 "tools": tools,
       2063                 "top_logprobs": top_logprobs,
       2064                 "top_p": top_p,
       2065                 "user": user,
       2066                 "web_search_options": web_search_options,
       2067             },
       2068             completion_create_params.CompletionCreateParamsStreaming
       2069             if stream
       2070             else completion_create_params.CompletionCreateParamsNonStreaming,
       2071         ),
       2072         options=make_request_options(
       2073             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
       2074         ),
       2075         cast_to=ChatCompletion,
       2076         stream=stream or False,
       2077         stream_cls=AsyncStream[ChatCompletionChunk],
       2078     )


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1805, in AsyncAPIClient.post(self, path, cast_to, body, files, options, stream, stream_cls)
       1791 async def post(
       1792     self,
       1793     path: str,
       (...)   1800     stream_cls: type[_AsyncStreamT] | None = None,
       1801 ) -> ResponseT | _AsyncStreamT:
       1802     opts = FinalRequestOptions.construct(
       1803         method="post", url=path, json_data=body, files=await async_to_httpx_files(files), **options
       1804     )
    -> 1805     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1495, in AsyncAPIClient.request(self, cast_to, options, stream, stream_cls, remaining_retries)
       1492 else:
       1493     retries_taken = 0
    -> 1495 return await self._request(
       1496     cast_to=cast_to,
       1497     options=options,
       1498     stream=stream,
       1499     stream_cls=stream_cls,
       1500     retries_taken=retries_taken,
       1501 )


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1585, in AsyncAPIClient._request(self, cast_to, options, stream, stream_cls, retries_taken)
       1583 if remaining_retries > 0 and self._should_retry(err.response):
       1584     await err.response.aclose()
    -> 1585     return await self._retry_request(
       1586         input_options,
       1587         cast_to,
       1588         retries_taken=retries_taken,
       1589         response_headers=err.response.headers,
       1590         stream=stream,
       1591         stream_cls=stream_cls,
       1592     )
       1594 # If the response is streamed then we need to explicitly read the response
       1595 # to completion before attempting to access the response text.
       1596 if not err.response.is_closed:


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1632, in AsyncAPIClient._retry_request(self, options, cast_to, retries_taken, response_headers, stream, stream_cls)
       1628 log.info("Retrying request to %s in %f seconds", options.url, timeout)
       1630 await anyio.sleep(timeout)
    -> 1632 return await self._request(
       1633     options=options,
       1634     cast_to=cast_to,
       1635     retries_taken=retries_taken + 1,
       1636     stream=stream,
       1637     stream_cls=stream_cls,
       1638 )


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1585, in AsyncAPIClient._request(self, cast_to, options, stream, stream_cls, retries_taken)
       1583 if remaining_retries > 0 and self._should_retry(err.response):
       1584     await err.response.aclose()
    -> 1585     return await self._retry_request(
       1586         input_options,
       1587         cast_to,
       1588         retries_taken=retries_taken,
       1589         response_headers=err.response.headers,
       1590         stream=stream,
       1591         stream_cls=stream_cls,
       1592     )
       1594 # If the response is streamed then we need to explicitly read the response
       1595 # to completion before attempting to access the response text.
       1596 if not err.response.is_closed:


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1632, in AsyncAPIClient._retry_request(self, options, cast_to, retries_taken, response_headers, stream, stream_cls)
       1628 log.info("Retrying request to %s in %f seconds", options.url, timeout)
       1630 await anyio.sleep(timeout)
    -> 1632 return await self._request(
       1633     options=options,
       1634     cast_to=cast_to,
       1635     retries_taken=retries_taken + 1,
       1636     stream=stream,
       1637     stream_cls=stream_cls,
       1638 )


    File c:\Users\tzehl\Documents\programming\promptolution\.venv\Lib\site-packages\openai\_base_client.py:1600, in AsyncAPIClient._request(self, cast_to, options, stream, stream_cls, retries_taken)
       1597         await err.response.aread()
       1599     log.debug("Re-raising status error")
    -> 1600     raise self._make_status_error_from_response(err.response) from None
       1602 return await self._process_response(
       1603     cast_to=cast_to,
       1604     options=options,
       (...)   1608     retries_taken=retries_taken,
       1609 )


    RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o-mini in organization org-3DmWJfR4tphuKTSzcsMB3vHF on requests per min (RPM): Limit 500, Used 500, Requested 1. Please try again in 120ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'requests', 'param': None, 'code': 'rate_limit_exceeded'}}



```python
prompts
```
