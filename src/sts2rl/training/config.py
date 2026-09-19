"""Serializable configuration and counters for repeatable training runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import torch

from sts2rl.agents import MAX_STATE_REFRESHES, PPOConfig
from sts2rl.encoder import EncoderConfig
from sts2rl.env import DEFAULT_ACTION_DELAY_SECONDS, ResetSpec


# Consecutive failed episodes before a client is considered gone rather than
# unlucky.  A crashed game should cost its own episodes, not the whole job, but
# a client that never comes back must stop consuming the episode budget.
MAX_EPISODE_FAILURES = 3

# Fixed run seeds, cycled one per episode.
#
# A fresh random run every episode puts map layout, card rewards, shops, and
# the enemy sequence into the return, where the critic can only ever predict
# their average -- everything seed-specific lands in the advantage as noise,
# and "mean return over the last N episodes" mixes policy improvement with
# draw luck.  A fixed pool makes that number comparable across checkpoints.
#
# A pool, not one seed: a single seed is memorized as an action sequence.  The
# holdout seeds are never trained on, so evaluating on them is what separates
# "learned to climb" from "learned these maps".
#
# Size is the dial between those two failures, and it is the only one that
# matters here.  Twelve seeds against the ~1500 episodes a run affords is 125
# plays of each map, which is enough to memorize the early floors; an unbounded
# pool plays each map once and spends the whole variance budget on draw luck.
# A hundred and fifty sits between them -- roughly ten plays of each -- and
# keeps the pool a fixed *distribution*, so a block mean stays comparable while
# per-seed memorization stops paying.
#
# These are real seeds, not invented ones: the game writes every finished run
# to %APPDATA%\SlayTheSpire2\**\saves\history\*.run as plaintext JSON, and
# the seeds of `game_mode: standard` runs are exactly what the game generates
# for itself.  9664 distinct ones were available; 180 were sampled and split.
# Do not hand-write seeds from a guessed alphabet.
DEFAULT_SEED_POOL = (
    "0357NDTC82", "066ET8NQLE", "07ZJQVNVL7", "0PCBECVTPL",
    "0SZ8ECRZ24", "0YGUCN8B3E", "0ZTT4AKVKQ", "110DDJMHPW",
    "137NFE5RG8", "144HTUX89Q", "1DGAKJS6N1", "1TG6C8U3N8",
    "2EU9F6SKEW", "2GZWK3AK07", "2RX76H6U40", "2WCXDRCQEK",
    "30MCWJY3ZW", "35SA6UP02F", "3EH64E978B", "3PANEGKLP8",
    "4E42J8772G", "4LUSRA08AA", "4UEVB4R020", "4WZYR2MXCA",
    "519KRCMJ1E", "5NMSDLXWL3", "5XH4KM54TQ", "5XV3DVMT54",
    "61GF5AVUQZ", "6S8W3SPCTK", "6SH31VZ0FJ", "6YUZ71TMFG",
    "71V1L4Y4AY", "73RAAPKKBU", "7FUMQ5U383", "7Q2BCFGE9J",
    "7U0Z1N999B", "80P165FWJK", "82VGM1L9X3", "85DZ31Z4JN",
    "86B0C37772", "8JZJ52Z0JD", "8W42J8N8BK", "8ZNZHNW39P",
    "92HLLQAXJX", "92S7VV54T8", "9BKMGETU32", "9ECTYSNSM9",
    "AB48KLTYDN", "AK43ULL3WS", "AKVPQYZBW0", "AY2KPZ2Y4F",
    "AZ7C354ZGG", "BE373Z71WC", "C44HJ097MA", "C5XTQ8N51B",
    "C6DV53VPJR", "CBDBHTM9M4", "CCV1BUE8JM", "CNL0342G9S",
    "CQ3PNLJZ4G", "DHLVJZ6EGF", "DM6RLN1NVV", "DWTS7PY5KL",
    "ET5LNZFVZU", "EU058TPXR2", "EXL7VX6GSF", "FHHFSB5W4C",
    "FKKT3Z53KN", "G5BKM8N1PE", "GTQKW4KCAH", "H14T0C5F9B",
    "H18EBU8G0C", "HD7R5MKQ2V", "J10MF934VG", "J3Q3TS7YQC",
    "JEQSXL4XVT", "JZ5N6AMTLB", "K678KQVS5F", "K8J1X2YT02",
    "KNAU1ZSPHB", "L5H8DCJH35", "L9FKQKD6G4", "LY016TGPH9",
    "M68VC3WXK6", "M6T3ATQ75E", "MCC3YA2MU5", "MJX3TR3ABP",
    "MKM92CSWXS", "MLUX0HMCMC", "MTD016VJZ6", "MVS24QNA90",
    "N4TEY69R4W", "NSAVTMQY28", "NYBVL26E3R", "NYQARG8CDA",
    "PEX21CB8CY", "PFFQ609NM0", "PTN3SALX8A", "PUHV5ZWXW6",
    "Q9DQCF87D5", "QT2X5DJYGY", "QU1EDZHZW2", "QVRPSHQCX1",
    "R0QAH22FT7", "RDFHM3K1TP", "RLD3D7R8S7", "RPQCVYAC73",
    "S0YXM0ZF7E", "S3D2UEQ2SA", "S7BJ2HE0N6", "SAQSERR9W6",
    "SEVAAV3J4H", "T0PUBZJ5J3", "TS3YZ85RYN", "TY6M3ZKZ7N",
    "U5YMYR3EDP", "UAQPVEC9B1", "UBQ0U1KPP2", "UE5594L22L",
    "UEBM9KYPJ1", "UPKK579N2X", "UYRE9J3ZCW", "V1WJ9EMZ3B",
    "V6JVY2SB2B", "VKPFDM8N0H", "VKYQ6RHFQY", "W3A8MC7UZ5",
    "WAQTZF819T", "WBRYST09GL", "WJQPTU9TKV", "WK11RMT05Q",
    "X0PQQB73K7", "X1D85PKAN3", "X6XDWGGYK2", "X8BB1PBC3H",
    "X96RX31CM1", "XMATWPFYRY", "XPTWPYU48C", "XRY1E24VDR",
    "XSNZH9Q4QL", "Y8FNXHLE1Y", "YATQ6AS8J0", "YB1FN6Y4KL",
    "YM58Q1TE7Z", "YMQ4BC2NH5", "YX5MBLDEZX", "ZL6K7D783Q",
    "ZQC7S0WBNA", "ZUVVK2SC7B",
)
DEFAULT_HOLDOUT_SEEDS = (
    "1HV0TDQF9C", "1PC2R1GS0T", "2022RDHNEP", "89WSDV9QQG",
    "8VACK0N352", "92MZ92WNYF", "AC64N3DSJ8", "CNFN4QK80E",
    "DD4ZL96VCV", "DG979J1U01", "DHNGF9Y8D2", "ETDA4LAPWY",
    "FEGQDD9EGJ", "H2T77H5C88", "H8BHDZSEB4", "M0VG0FRGBS",
    "MBCEEP4024", "MBXM1CC2L6", "N7QUJJ2AEA", "QU54MPMGQE",
    "SAUP1VW3LY", "SRG0QRCVQ7", "UMRVMH92DS", "UUP0VM3LZA",
    "VCDTTUXZXK", "WQ9865E3X6", "Y03PUR11PD", "YQ26XB9RDL",
    "Z69Q62VL4J", "ZWRRHMF29A",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Operational settings that do not define model tensor shapes."""

    total_episodes: int = 100
    # Optimizer updates between checkpoints, not episodes: an update is a
    # fixed rollout of transitions, while episode length here grows about
    # fourfold over a run, so counting episodes silently changes the interval.
    checkpoint_every: int = 10
    max_steps_per_episode: int = 10_000
    max_state_refreshes: int = MAX_STATE_REFRESHES
    max_episode_failures: int = MAX_EPISODE_FAILURES
    base_url: str = "http://localhost:15526/api/v1"
    timeout: float = 20.0
    action_delay_seconds: float = DEFAULT_ACTION_DELAY_SECONDS
    ports: tuple[int, ...] = ()
    training_seeds: tuple[str, ...] = ()
    holdout_seeds: tuple[str, ...] = ()
    device: str = "cpu"
    torch_seed: int = 0
    run_dir: Path = Path("runs/default")
    tensorboard_enabled: bool = True
    tensorboard_flush_secs: int = 30
    # The behavior-cloning artifact the encoder started from, or None for
    # random weights.  Recorded rather than only applied: two runs identical in
    # every other field are different experiments if one started from cloned
    # weights, and nothing else in the run directory would say so.
    init_encoder: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "total_episodes",
            "checkpoint_every",
            "max_steps_per_episode",
            "max_episode_failures",
            "tensorboard_flush_secs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_state_refreshes, bool)
            or not isinstance(self.max_state_refreshes, int)
            or self.max_state_refreshes < 0
        ):
            raise ValueError("max_state_refreshes must be a non-negative integer")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(self.action_delay_seconds, bool) or not isinstance(
            self.action_delay_seconds, (int, float)
        ):
            raise TypeError("action_delay_seconds must be a number")
        if self.action_delay_seconds < 0:
            raise ValueError("action_delay_seconds must not be negative")
        if isinstance(self.torch_seed, bool) or not isinstance(self.torch_seed, int):
            raise ValueError("torch_seed must be an integer")
        if not isinstance(self.base_url, str):
            raise TypeError("base_url must be a string")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not isinstance(self.tensorboard_enabled, bool):
            raise TypeError("tensorboard_enabled must be a boolean")
        if self.init_encoder is not None and (
            not isinstance(self.init_encoder, str) or not self.init_encoder
        ):
            raise ValueError("init_encoder must be a non-empty path or None")
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid torch device: {self.device!r}") from exc
        ports = tuple(self.ports)
        object.__setattr__(self, "ports", ports)
        if any(
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
            for port in ports
        ):
            raise ValueError("ports must be integers between 1 and 65535")
        if len(set(ports)) != len(ports):
            raise ValueError("ports must not repeat; one client per port")
        for name in ("training_seeds", "holdout_seeds"):
            seeds = tuple(getattr(self, name))
            object.__setattr__(self, name, seeds)
            if any(not isinstance(seed, str) or not seed for seed in seeds):
                raise ValueError(f"{name} must be non-empty strings")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must not repeat a seed")
        overlap = set(self.training_seeds) & set(self.holdout_seeds)
        if overlap:
            raise ValueError(
                "holdout seeds must never be trained on; both pools list "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "run_dir", Path(self.run_dir))

    def validate_runtime_device(self) -> torch.device:
        """Resolve the configured device and reject unavailable CUDA devices."""
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
        return device

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["run_dir"] = str(self.run_dir)
        result["ports"] = list(self.ports)
        result["training_seeds"] = list(self.training_seeds)
        result["holdout_seeds"] = list(self.holdout_seeds)
        return result

    def client_base_urls(self) -> tuple[str, ...]:
        """Return one base URL per client, in configuration order.

        Clients are added as ports, so the configured ``base_url`` supplies the
        scheme, host, and path and each port replaces its port component. With
        no ports it is the single client, exactly as before.
        """
        if not self.ports:
            return (self.base_url,)
        parsed = urlsplit(self.base_url)
        host = parsed.hostname or "localhost"
        return tuple(
            urlunsplit(
                (
                    parsed.scheme,
                    f"{host}:{port}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
            for port in self.ports
        )

    def seed_for_episode(self, completed_episodes: int) -> str | None:
        """Return the seed the next episode should run, cycling the pool.

        The cursor is derived from the episode counter rather than stored, so
        a resumed run continues the cycle instead of restarting it -- which
        would otherwise re-train the same few seeds and starve the rest.
        """
        if not self.training_seeds:
            return None
        return self.training_seeds[completed_episodes % len(self.training_seeds)]

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingConfig:
        """Restore configuration from its validated JSON representation."""
        data = dict(values)
        if "run_dir" in data:
            data["run_dir"] = Path(str(data["run_dir"]))
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrainingPlan:
    """Complete immutable configuration of one training experiment."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reset: ResetSpec = field(default_factory=ResetSpec)

    def __post_init__(self) -> None:
        if self.reset.character not in range(5):
            raise ValueError("reset character must be between 0 and 4")
        if self.reset.game_mode == "standard" and self.reset.run_seed is not None:
            raise ValueError("run_seed is only supported for custom or daily runs")
        if self.training.training_seeds and self.reset.game_mode != "custom":
            raise ValueError(
                "a seed pool needs the custom-run screen, which is the only one "
                "that accepts a seed; pass --game-mode custom"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete plan as JSON-compatible primitives."""
        return {
            "training": self.training.to_dict(),
            "encoder": asdict(self.encoder),
            "ppo": asdict(self.ppo),
            "reset": asdict(self.reset),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingPlan:
        """Restore a complete plan and rerun every component's validation."""
        training = _mapping(values.get("training"), "training")
        encoder = _mapping(values.get("encoder"), "encoder")
        ppo = _mapping(values.get("ppo"), "ppo")
        reset = _mapping(values.get("reset"), "reset")
        return cls(
            training=TrainingConfig.from_dict(training),
            encoder=EncoderConfig(**encoder),  # type: ignore[arg-type]
            ppo=PPOConfig(**ppo),  # type: ignore[arg-type]
            reset=ResetSpec(**reset),  # type: ignore[arg-type]
        )


@dataclass
class TrainingState:
    """Mutable counters persisted across process restarts."""

    completed_episodes: int = 0
    environment_steps: int = 0
    optimizer_updates: int = 0

    def __post_init__(self) -> None:
        for name in (
            "completed_episodes",
            "environment_steps",
            "optimizer_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        """Return counters as checkpoint-safe primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingState:
        """Restore and validate persisted counters."""
        return cls(**dict(values))  # type: ignore[arg-type]


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value
