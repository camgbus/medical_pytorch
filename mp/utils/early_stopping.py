from mp.eval.result import Result


class EarlyStopping:
    r"""
    Stops the training early if the metric score don't improve within a given time period.
    """

    def __init__(self, patience: int, metric: str, ds_names: list, metric_min_delta: float = 33e-4):
        r"""
        Args:
            patience (int): the number of checks without improvement allowed, before stopping training
            metric (str):  a metric name, which can be used as keys in a Results object
            ds_names (list): a list of datasets names, which can be used as keys in a Results object
            metric_min_delta: minimum metric score increase required to consider the change as an improvement
        """
        self.patience = patience
        self.metric = metric
        self.ds_names = ds_names
        self.metric_min_delta = metric_min_delta
        self.no_improvement_counter = 0
        self.best_scores = []

    def reset_counter(self):
        self.no_improvement_counter = 0

    def reset(self):
        self.no_improvement_counter = 0
        self.best_scores = []

    def check_results(self, results: Result, epoch: int) -> bool:
        r"""Given a Result object checks whether the metric scores have improved"""
        assert self.metric in results.results and epoch in results.results[self.metric], \
            "metric and/or epoch not found in Results object"

        results = results.results[self.metric][epoch]

        # No best scores yet
        if not self.best_scores:
            self.best_scores = [results[name] for name in self.ds_names]
            return True

        # Otherwise check for score changes
        improvement = False  # Flag indicating whether at least one score has improved
        for idx, ds_name in enumerate(self.ds_names):
            score = results[ds_name]
            # Improvement: update score and flag, resetting counter
            if score >= self.best_scores[idx] + self.metric_min_delta:
                self.best_scores[idx] = score
                improvement = True
                self.no_improvement_counter = 0

        # Updating the counter accordingly
        if not improvement:
            self.no_improvement_counter += 1

        return self.no_improvement_counter <= self.patience
