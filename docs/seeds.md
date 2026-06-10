# Seeds

Evaluation seed generation is designed for fair checkpoint comparison.

For `episodes = n` and `clients = m`, STS2RL generates `n * m` seeds once at the
start of evaluation. Each client receives `n` seeds, and that assignment is used
for every checkpoint.

With `--episodes 3 --client-port A --client-port B --seed S`:

```text
client-1=[S, S_2, S_3]
client-2=[S_4, S_5, S_6]
```

Without `--seed`, random 10-character seeds are generated.

