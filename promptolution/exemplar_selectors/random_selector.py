from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector


class RandomSelector(BaseExemplarSelector):
    def __init__(self, task, predictor):
        super().__init__()
        self.task = task
        self.predictor = predictor

    def select_exemplars(self, prompt, n_examples: int = 5):
        # use shape for evaluation st correct or incorrect can be identified
        examples = []
        while len(examples) < n_examples:
            score, seq = self.task.evaluate(prompt, self.predictor, n_samples=1, return_seq=True)
            if score == 1:
                examples.append(seq[0])
        prompt = "\n".join(examples + [prompt])

        return prompt
