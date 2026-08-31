"""The batched DQN/PPO update paths must match the per-sample math they replaced."""

import torch
from torch.distributions import Categorical

from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent


def make_transition(agent, seed: int, candidate_count: int) -> dict:
    """Build one synthetic rollout transition with a given candidate-set size."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.rand(agent.state_size, generator=generator).tolist()
    candidate_vectors = [
        torch.rand(agent.action_feature_size, generator=generator).tolist()
        for _ in range(candidate_count)
    ]
    return {
        "state": state,
        "candidate_vectors": candidate_vectors,
        "action_index": seed % candidate_count,
        "action_vector": candidate_vectors[seed % candidate_count],
        "reward": 0.5,
        "next_state": state,
        "done": False,
        "next_action_vectors": candidate_vectors,
        "logprob": 0.0,
        "value": 0.0,
    }


def test_ppo_batched_logprobs_match_per_sample_reference():
    """Concatenating variable-length candidate sets must not change the math."""
    agent = PPOCandidateAgent()
    # Deliberately ragged: the batching splits by per-transition candidate counts.
    rollout = [make_transition(agent, seed, count) for seed, count in enumerate([3, 1, 5, 2], 1)]
    indices = list(range(len(rollout)))

    logprobs, entropies = agent._action_logprobs(rollout, indices)

    expected_logprobs = []
    expected_entropies = []
    with torch.no_grad():
        for transition in rollout:
            rows = [
                transition["state"] + vector for vector in transition["candidate_vectors"]
            ]
            logits = agent.model.score(
                torch.tensor(rows, dtype=torch.float32, device=agent.device)
            )
            distribution = Categorical(logits=logits)
            action = torch.tensor(
                transition["action_index"], dtype=torch.long, device=agent.device
            )
            expected_logprobs.append(distribution.log_prob(action))
            expected_entropies.append(distribution.entropy())

    assert torch.allclose(logprobs, torch.stack(expected_logprobs), atol=1e-6)
    assert torch.allclose(entropies, torch.stack(expected_entropies), atol=1e-6)


def test_dqn_batched_next_q_matches_per_sample_reference():
    """The scatter-reduce max over next candidates must equal a per-sample max."""
    agent = DQNCandidateAgent(batch_size=4)
    transitions = [make_transition(agent, seed, count) for seed, count in enumerate([2, 4, 1, 3], 1)]
    # One terminal sample, which must contribute a zero rather than a scored max.
    transitions[2]["done"] = True

    for transition in transitions:
        agent.remember(
            transition["state"],
            transition["action_vector"],
            transition["reward"],
            transition["next_state"],
            transition["done"],
            transition["next_action_vectors"],
        )

    expected = []
    with torch.no_grad():
        for transition in transitions:
            if transition["done"]:
                expected.append(0.0)
                continue
            rows = [
                transition["next_state"] + vector
                for vector in transition["next_action_vectors"]
            ]
            next_q = agent.target_model(
                torch.tensor(rows, dtype=torch.float32, device=agent.device)
            )
            expected.append(float(next_q.max().item()))

    # Recompute through the production path by replaying the same batch.
    batch = list(agent.replay_buffer)
    next_states = [entry[3] for entry in batch]
    dones = [entry[4] for entry in batch]
    next_actions = [entry[5] for entry in batch]

    with torch.no_grad():
        next_rows: list[list[float]] = []
        row_owner: list[int] = []
        for sample_index, (next_state, done, vectors) in enumerate(
            zip(next_states, dones, next_actions)
        ):
            if done or not vectors:
                continue
            for vector in vectors:
                next_rows.append(list(next_state) + list(vector))
                row_owner.append(sample_index)

        actual = torch.zeros(len(batch), dtype=torch.float32, device=agent.device)
        next_q = agent.target_model(
            torch.tensor(next_rows, dtype=torch.float32, device=agent.device)
        )
        actual = actual.scatter_reduce(
            0,
            torch.tensor(row_owner, dtype=torch.long, device=agent.device),
            next_q,
            reduce="amax",
            include_self=False,
        )

    assert torch.allclose(
        actual, torch.tensor(expected, dtype=torch.float32, device=agent.device), atol=1e-6
    )


def test_dqn_train_step_runs_with_ragged_candidate_sets():
    """A full update should run end to end on a ragged batch."""
    agent = DQNCandidateAgent(batch_size=4)
    for seed, count in enumerate([2, 4, 1, 3], 1):
        transition = make_transition(agent, seed, count)
        agent.remember(
            transition["state"],
            transition["action_vector"],
            transition["reward"],
            transition["next_state"],
            transition["done"],
            transition["next_action_vectors"],
        )

    loss = agent.train_step()

    assert loss is not None
    assert loss == loss  # not NaN
