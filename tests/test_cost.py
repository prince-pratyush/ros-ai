import pytest

from rosai.cost import CostCeilingError, estimate_cost, guard

# Hand sums of the 19 Oct 2026 Modal base rates.
# 10800 * 0.000694 + 8 * 10800 * 0.0000131 + 32 * 10800 * 0.00000222
A100_THREE_HOURS = 9.394272
# 10800 * 0.001097 + the same CPU and memory
H100_THREE_HOURS = 13.746672


def test_three_hour_preemptible_a100_matches_the_rate_table(capsys):
    usd = guard(
        gpu="A100-80GB",
        seconds=10800,
        cpu_cores=8,
        memory_gib=32,
        max_usd=12,
        max_gpu_hours=3,
    )
    assert usd == A100_THREE_HOURS
    assert estimate_cost(
        gpu="A100-80GB", seconds=10800, cpu_cores=8, memory_gib=32
    ) == A100_THREE_HOURS
    assert "9.394272" in capsys.readouterr().out


def test_h100_costs_more_and_a_twelve_dollar_ceiling_refuses_it():
    h100 = estimate_cost(gpu="H100", seconds=10800, cpu_cores=8, memory_gib=32)
    assert h100 == H100_THREE_HOURS
    assert h100 > A100_THREE_HOURS
    with pytest.raises(CostCeilingError) as exc:
        guard(
            gpu="H100",
            seconds=10800,
            cpu_cores=8,
            memory_gib=32,
            max_usd=12,
            max_gpu_hours=3,
        )
    assert "12" in str(exc.value)


def test_gpu_hour_ceiling_refuses_a_run_that_is_still_under_the_dollar_cap():
    with pytest.raises(CostCeilingError) as exc:
        guard(
            gpu="A100-80GB",
            seconds=14400,
            cpu_cores=8,
            memory_gib=32,
            max_usd=1000,
            max_gpu_hours=3,
        )
    assert "gpu hours" in str(exc.value)


def test_one_retry_prices_two_attempts_of_a_lean_a100(capsys):
    # 2 * 1200 * (0.000694 + 8 * 0.0000131 + 32 * 0.00000222)
    usd = guard(
        gpu="A100-80GB",
        seconds=1200,
        cpu_cores=8,
        memory_gib=32,
        max_usd=3,
        max_gpu_hours=1,
        retries=1,
    )
    assert usd == 2.087616
    assert "2.087616" in capsys.readouterr().out


def test_three_retries_are_refused_when_four_attempts_pass_the_dollar_ceiling():
    with pytest.raises(CostCeilingError) as exc:
        guard(
            gpu="A100-80GB",
            seconds=1200,
            cpu_cores=8,
            memory_gib=32,
            max_usd=3,
            max_gpu_hours=4,
            retries=3,
        )
    assert "4.175232" in str(exc.value)


def test_retry_attempts_count_against_the_gpu_hour_ceiling():
    with pytest.raises(CostCeilingError) as exc:
        guard(
            gpu="A100-80GB",
            seconds=1200,
            cpu_cores=8,
            memory_gib=32,
            max_usd=100,
            max_gpu_hours=1,
            retries=3,
        )
    assert "gpu hours" in str(exc.value)
