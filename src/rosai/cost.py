"""Estimate a Modal run from the published base rates and refuse a ceiling breach."""

from decimal import Decimal

# Base (preemptible) prices from https://modal.com/pricing on 19 Oct 2026.
# Non-preemptible execution is 3x these prices, so callers stay on the base rate.
_GPU_USD_PER_SECOND = {
    "A100-80GB": Decimal("0.000694"),
    "H100": Decimal("0.001097"),
}
_CPU_USD_PER_CORE_SECOND = Decimal("0.0000131")
_MEMORY_USD_PER_GIB_SECOND = Decimal("0.00000222")


class CostCeilingError(Exception):
    """A launch would exceed max_usd or max_gpu_hours."""


def estimate_cost(*, gpu: str, seconds: int, cpu_cores: int, memory_gib: int) -> float:
    duration = Decimal(seconds)
    gpu_usd = duration * _GPU_USD_PER_SECOND[gpu]
    cpu_usd = Decimal(cpu_cores) * duration * _CPU_USD_PER_CORE_SECOND
    memory_usd = Decimal(memory_gib) * duration * _MEMORY_USD_PER_GIB_SECOND
    return float(gpu_usd + cpu_usd + memory_usd)


def guard(
    *,
    gpu: str,
    seconds: int,
    cpu_cores: int,
    memory_gib: int,
    max_usd: float,
    max_gpu_hours: float,
) -> float:
    usd = estimate_cost(
        gpu=gpu, seconds=seconds, cpu_cores=cpu_cores, memory_gib=memory_gib
    )
    gpu_hours = seconds / 3600
    print(
        f"estimate ${usd:.6f} on preemptible {gpu} "
        f"for {seconds}s, {cpu_cores} cores, {memory_gib} GiB"
    )
    reasons: list[str] = []
    if usd > max_usd:
        reasons.append(f"${usd:.6f} exceeds ${max_usd:.2f}")
    if gpu_hours > max_gpu_hours:
        reasons.append(f"{gpu_hours:g} gpu hours exceeds {max_gpu_hours:g}")
    if reasons:
        raise CostCeilingError(f"{gpu}: " + "; ".join(reasons))
    return usd
