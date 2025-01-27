import math
from dataclasses import asdict, dataclass, field

import torch


@dataclass
class SimulationHyperparameters:
    kappa: float = 0.5
    bond_penalty: float = 1.0
    total_epoch_emission: float = 100.0
    validator_emission_ratio: float = 0.41
    total_subnet_stake: float = 1_000_000.0
    consensus_precision: int = 100_000


@dataclass
class YumaParams:
    bond_alpha: float = 0.1
    liquid_alpha: bool = False
    alpha_high: float = 0.9
    alpha_low: float = 0.7
    decay_rate: float = 0.1
    capacity_alpha: float = 0.1
    override_consensus_high: float | None = None
    override_consensus_low: float | None = None


@dataclass
class YumaConfig:
    simulation: SimulationHyperparameters = field(
        default_factory=SimulationHyperparameters
    )
    yuma_params: YumaParams = field(default_factory=YumaParams)

    def __post_init__(self):
        # Flatten fields for direct access
        simulation_dict = asdict(self.simulation)
        yuma_params_dict = asdict(self.yuma_params)

        for key, value in simulation_dict.items():
            setattr(self, key, value)

        for key, value in yuma_params_dict.items():
            setattr(self, key, value)


@dataclass(frozen=True)
class YumaSimulationNames:
    YUMA_RUST: str = "Yuma 0 (subtensor)"
    YUMA: str = "Yuma 1 (paper)"
    YUMA_LIQUID: str = "Yuma 1 (paper) - liquid alpha on"
    YUMA2: str = "Yuma 2 (Adrian-Fish)"
    YUMA3: str = "Yuma 3 (Rhef)"
    YUMA31: str = "Yuma 3.1 (Rhef+reset)"
    YUMA32: str = "Yuma 3.2 (Rhef+conditional)"
    YUMA4: str = "Yuma 4 (Rhef+relative bonds)"
    YUMA4_LIQUID: str = "Yuma 4 (Rhef+relative bonds) - liquid alpha on"


def YumaRust(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | str | float]:
    """
    Currently implemented Subtensor Yuma function.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1], dtype=torch.float64)

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    B = S.view(-1, 1) * W_clipped
    B_sum = B.sum(dim=0)
    B = B / (B_sum + 1e-6)
    B = torch.nan_to_num(B)

    a = b = torch.tensor(float('nan'))
    bond_alpha = config.bond_alpha
    if config.liquid_alpha:
        consensus_high = (
            config.override_consensus_high
            if config.override_consensus_high is not None
            else C.quantile(0.75)
        )
        consensus_low = (
            config.override_consensus_low
            if config.override_consensus_low is not None
            else C.quantile(0.25)
        )

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (
            math.log(1 / config.alpha_high - 1) - math.log(1 / config.alpha_low - 1)
        ) / (consensus_low - consensus_high)
        b = math.log(1 / config.alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, config.alpha_low, config.alpha_high)

    if B_old is not None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B.clone()

    B_ema_sum = B_ema.sum(dim=0)
    B_ema = B_ema / (B_ema_sum + 1e-6)
    B_ema = torch.nan_to_num(B_ema)

    # === Dividend Calculation===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "validator_bond": B,
        "validator_ema_bond": B_ema,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "bond_alpha": bond_alpha,
        "alpha_a": a,
        "alpha_b": b,
    }


def Yuma(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | None | float]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - config.bond_penalty) * W + config.bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = torch.tensor(float('nan'))
    bond_alpha = config.bond_alpha
    if config.liquid_alpha:
        consensus_high = (
            config.override_consensus_high
            if config.override_consensus_high is not None
            else C.quantile(0.75)
        )
        consensus_low = (
            config.override_consensus_low
            if config.override_consensus_low is not None
            else C.quantile(0.25)
        )

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (
            math.log(1 / config.alpha_high - 1) - math.log(1 / config.alpha_low - 1)
        ) / (consensus_low - consensus_high)
        b = math.log(1 / config.alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, config.alpha_low, config.alpha_high)

    if B_old is not None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "weight_for_bond": W_b,
        "validator_bond": B,
        "validator_ema_bond": B_ema,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "bond_alpha": bond_alpha,
        "alpha_a": a,
        "alpha_b": b,
    }


def Yuma2(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | None | float]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    if W_prev is None:
        W_prev = W

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W_prev, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - config.bond_penalty) * W_prev + config.bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = torch.tensor(float('nan'))
    bond_alpha = config.bond_alpha
    if config.liquid_alpha:
        consensus_high = (
            config.override_consensus_high
            if config.override_consensus_high is not None
            else C.quantile(0.75)
        )
        consensus_low = (
            config.override_consensus_low
            if config.override_consensus_low is not None
            else C.quantile(0.25)
        )

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (
            math.log(1 / config.alpha_high - 1) - math.log(1 / config.alpha_low - 1)
        ) / (consensus_low - consensus_high)
        b = math.log(1 / config.alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, config.alpha_low, config.alpha_high)

    if B_old is not None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "weight_for_bond": W_b,
        "validator_bond": B,
        "validator_ema_bond": B_ema,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "bond_alpha": bond_alpha,
        "alpha_a": a,
        "alpha_b": b,
    }


def Yuma3(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
    maxint: int = 2**64 - 1,
) -> dict[str, torch.Tensor | None | float]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    capacity = S * maxint

    # Compute Remaining Capacity
    capacity_per_bond = S.unsqueeze(1) * maxint
    remaining_capacity = capacity_per_bond - B_old
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Compute Purchase Capacity
    capacity_alpha = (config.capacity_alpha * capacity).unsqueeze(1)
    purchase_capacity = torch.min(capacity_alpha, remaining_capacity)

    # Allocate Purchase to Miners
    purchase = purchase_capacity * W

    # Update Bonds with Decay and Purchase
    decay = 1 - config.decay_rate
    B = decay * B_old + purchase
    B = torch.min(B, capacity_per_bond)  # Enforce capacity constraints

    # === Validator reward ===
    D = (B * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
    }


def Yuma4(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | None | float]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Liquid Alpha Adjustment ===
    a = b = torch.tensor(float('nan'))
    bond_alpha = config.bond_alpha
    if config.liquid_alpha:
        consensus_high = (
            config.override_consensus_high
            if config.override_consensus_high is not None
            else C.quantile(0.75)
        )
        consensus_low = (
            config.override_consensus_low
            if config.override_consensus_low is not None
            else C.quantile(0.25)
        )

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (
            math.log(1 / config.alpha_high - 1) - math.log(1 / config.alpha_low - 1)
        ) / (consensus_low - consensus_high)
        b = math.log(1 / config.alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, config.alpha_low, config.alpha_high)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    B_decayed = B_old * (1 - bond_alpha)
    remaining_capacity = 1.0 - B_decayed
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Each validator can increase bonds by at most bond_alpha per epoch towards the cap
    purchase_increment = (
        bond_alpha * W
    )  # Validators allocate their purchase across miners based on weights
    # Ensure that purchase does not exceed remaining capacity
    purchase = torch.min(purchase_increment, remaining_capacity)

    B = B_decayed + purchase
    B = torch.clamp(B, max=1.0)

    # === Dividends Calculation ===
    total_bonds_per_validator = (B * I).sum(dim=1)  # Sum over miners for each validator
    D = S * total_bonds_per_validator  # Element-wise multiplication

    # Normalize dividends
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
    }
