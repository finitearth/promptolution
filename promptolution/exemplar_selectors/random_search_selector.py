"""Random search exemplar selector."""

from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector


class RandomSearchSelector(BaseExemplarSelector):
    """A selector that uses random search to find the best set of exemplars.

    This class implements a strategy that generates multiple sets of random examples,
    evaluates their performance, and selects the best performing set.
    """

    def select_exemplars(self, prompt: str, n_trials: int = 5) -> str:
        """Select exemplars using a random search strategy.

        This method generates multiple sets of random examples, evaluates their performance
        when combined with the original prompt, and returns the best performing set.

        Args:
            prompt (str): The input prompt to base the exemplar selection on.
            n_trials (int, optional): The number of random trials to perform. Defaults to 5.

        Returns:
            str: The best performing prompt, which includes the original prompt and the selected exemplars.
        """
        best_score = 0.0
        best_prompt = prompt

        for _ in range(n_trials):
            _, seq = self.task.evaluate(prompt, self.predictor, eval_strategy="subsample", return_seq=True)
            prompt_with_examples = "\n\n".join([prompt] + [seq.item()]) + "\n\n"
            # evaluate prompts as few shot prompt
            score, _ = self.task.evaluate(prompt_with_examples, self.predictor, eval_strategy="subsample")
            assert isinstance(score, float), f"Expected float, but got {type(score).__name__}"
            if score > best_score:
                best_score = score
                best_prompt = prompt_with_examples

        return best_prompt
