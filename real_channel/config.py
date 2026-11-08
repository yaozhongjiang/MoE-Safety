"""Public experiment configuration. Never includes victim routing or prompt labels."""
from dataclasses import asdict, dataclass


PRESETS = {
    "smoke": dict(n_train=2, n_test=2, steps=2, layers="12", seeds=1, epochs=10),
    "quick": dict(n_train=16, n_test=16, steps=4, layers="12", seeds=5, epochs=200),
    "paper": dict(n_train=150, n_test=150, steps=32, layers="all", seeds=5, epochs=200, shard_requests=1),
}


@dataclass
class Config:
    model_id: str = "AI-ModelScope/granite-3.0-1b-a400m-base"
    model_revision: str = "master"
    dtype: str = "bfloat16"
    tasks: str = "code"
    n_train: int = 16
    n_test: int = 16
    tokens: int = 32
    steps: int = 4
    layers: str = "12"
    seeds: int = 5
    epochs: int = 200
    sample_seed: int = 7
    shard_requests: int = 16
    repeats: int = 3
    benchmark_requests: int = 8
    E: int = 32
    K: int = 8
    L: int = 24

    def layer_ids(self):
        ids = list(range(self.L)) if self.layers == "all" else [int(v) for v in self.layers.split(",")]
        if not ids or len(set(ids)) != len(ids) or any(v < 0 or v >= self.L for v in ids):
            raise ValueError("layers 必须是 all 或互不重复的零起始层号")
        return sorted(ids)

    def validate(self):
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype 必须为 bfloat16 或 float16")
        if not 1 <= self.steps <= self.tokens <= 128:
            raise ValueError("需要 1 <= steps <= tokens <= 128")
        for key in ("n_train", "n_test", "seeds", "epochs", "shard_requests", "repeats", "benchmark_requests"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} 必须为正数")
        if min(self.n_train, self.n_test) < 2:
            raise ValueError("每类每个 split 至少两个样本")
        tasks = [v.strip() for v in self.tasks.split(",")]
        if not set(tasks) <= {"code", "medical", "nationality", "sensitive"} or len(set(tasks)) != len(tasks):
            raise ValueError("未知属性任务")
        self.tasks = ",".join(tasks)
        if not 1 <= self.K <= self.E or self.L < 1:
            raise ValueError("非法 MoE 维度")
        self.layer_ids()

    def public(self):
        # Config contains no prompt content, labels, or truth.
        return asdict(self)
