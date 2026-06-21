"""TRL critic_agent YAML helpers (keys unused by ``trl_loss``)."""

from __future__ import annotations

from typing import Any

_TRL_ALIASES = frozenset(
    {
        'trl',
        'state_transitive',
        'transitive_v_local_q',
    }
)

# DQC/IQL-only; ignored by TRL (``trl_loss`` never reads these).
TRL_UNUSED_CRITIC_KEYS: tuple[str, ...] = (
    'use_chunk_critic',
    'q_target_from_value',
    'kappa_b',
    'kappa_d',
    'distill_method',
    'implicit_backup_type',
)

# State-SPI / Q_Z keys — only used when ``lambda_q_local=0`` (V-SPI actor path).
TRL_ACTION_CHUNK_SPI_KEYS: tuple[str, ...] = (
    'state_spi_energy_type',
    'use_qz_critic',
    'lambda_qz_prod',
    'lambda_qz_tri',
    'qz_tau',
    'qz_use_transitive_backup',
)


def is_trl_critic_config(config: dict[str, Any]) -> bool:
    critic_type = str(config.get('critic_type', 'dqc')).lower()
    algorithm = str(config.get('algorithm', '')).lower()
    return critic_type in _TRL_ALIASES or algorithm in _TRL_ALIASES


def strip_trl_unused_critic_keys(config: dict[str, Any]) -> dict[str, Any]:
    for key in TRL_UNUSED_CRITIC_KEYS:
        if key in config:
            del config[key]
    return config


def strip_trl_action_chunk_spi_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Remove V-SPI/Q_Z knobs when training action-chunk TRL (``lambda_q_local>0``)."""
    if float(config.get('lambda_q_local', 0.0)) > 0.0:
        for key in TRL_ACTION_CHUNK_SPI_KEYS:
            config.pop(key, None)
    return config


def finalize_trl_critic_config(config: dict[str, Any]) -> dict[str, Any]:
    strip_trl_unused_critic_keys(config)
    strip_trl_action_chunk_spi_keys(config)
    return config


def trl_critic_agent_config(
    *,
    discount: float,
    lambda_q_local: float = 1.0,
    value_distance_weight_power: float = 0.0,
    max_goal_steps_from_env: bool = False,
    state_spi_energy_type: str = 'v_product',
    use_qz_critic: bool = False,
    value_goals: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Minimal ``critic_agent`` block for TRL sweep YAML generators."""
    cfg: dict[str, Any] = {
        'algorithm': 'trl',
        'critic_type': 'trl',
        'action_chunk_horizon': 5,
        'full_chunk_horizon': 25,
        'discount': float(discount),
        'tau_v': 0.7,
        'lambda_v_self': 1.0,
        'lambda_v_base': 1.0,
        'lambda_v_tri': 1.0,
        'value_base_horizon': 5,
        'value_transitive_reweight': True,
        'value_distance_weight_power': float(value_distance_weight_power),
        'lambda_q_local': float(lambda_q_local),
        'goal_representation': 'full',
        'max_goal_steps_from_env': bool(max_goal_steps_from_env),
        'clip_chunk_to_goal': True,
        'subgoal_value_bonus_type': 'transitive_product',
        'subgoal_value_log_eps': 1e-6,
        'subgoal_value_ratio_eps': 1e-3,
        'subgoal_value_ratio_clip': 5.0,
    }
    if float(lambda_q_local) <= 0.0:
        cfg['state_spi_energy_type'] = state_spi_energy_type
        cfg['use_qz_critic'] = use_qz_critic
    if value_goals:
        cfg.update(value_goals)
    cfg.update(extra)
    return cfg
