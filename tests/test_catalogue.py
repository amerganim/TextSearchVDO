"""Hardware probing and the model catalogue.

These are the pieces that decide what a given machine is offered. Getting them
wrong is quiet: somebody is handed a model their laptop cannot carry, or -
worse for anyone shipping this - a model whose licence forbids it, and nothing
says so until much later.
"""

from __future__ import annotations

import pytest

from tsv import catalogue
from tsv.catalogue import STAGES, ModelChoice, choices, fits, in_use, recommend, unshippable
from tsv.config import DEFAULT
from tsv.hardware import Accelerator, Machine, probe, total_ram_mb


# ---------- the probe ----------

def test_the_probe_reports_a_plausible_machine():
    machine = probe()
    assert machine.cpu_cores >= 1
    assert machine.ram_mb > 512, "physical memory could not be read at all"
    assert machine.os_name


def test_memory_reading_never_raises():
    """A probe that throws is worse than one that shrugs.

    It runs on every setup and every hardware page, on machines this has never
    seen. Returning 0 is a usable answer; an exception out of a capability
    check is not.
    """
    assert total_ram_mb() >= 0


def test_a_listed_accelerator_is_not_treated_as_a_usable_one():
    """`usable is None` means nobody looked.

    onnxruntime lists the providers its wheel was built with, not the ones
    this machine can run - a CUDA provider with no driver is listed and fails
    at session creation. Counting that as capable is how somebody is offered a
    model that cannot load.
    """
    untested = Accelerator("onnxruntime", "CUDA", name="GeForce RTX 4090", vram_mb=24576)
    machine = Machine(cpu_cores=8, ram_mb=32768, accelerators=(untested,))

    assert machine.usable_accelerators == ()
    assert machine.best is None
    assert machine.has_discrete_gpu is False
    assert "not yet tested" in machine.summary()
    assert "no GPU" not in machine.summary(), "claimed there is no GPU without checking"


def test_a_machine_with_nothing_says_no_gpu_plainly():
    machine = Machine(cpu_cores=4, ram_mb=8192)
    assert "no GPU" in machine.summary()


def test_an_integrated_gpu_loses_to_a_discrete_one():
    """The ordering that matters.

    On the baseline machine the integrated GPU was slower than the CPU on
    every model except the detector, so it must never outrank a real card.
    """
    integrated = Accelerator("openvino", "GPU", usable=True)
    discrete = Accelerator("onnxruntime", "CUDA", name="GeForce RTX 3060",
                           vram_mb=12288, usable=True)
    machine = Machine(8, 32768, accelerators=(integrated, discrete))

    assert integrated.kind == "integrated"
    assert discrete.kind == "discrete"
    assert machine.best is discrete
    assert machine.has_discrete_gpu is True


def test_an_unusable_accelerator_is_excluded_and_explains_itself():
    broken = Accelerator("onnxruntime", "CUDA", usable=False, note="no driver")
    machine = Machine(8, 16384, accelerators=(broken,))
    assert machine.best is None
    assert machine.untested_accelerators == ()


# ---------- the catalogue ----------

def test_every_stage_offers_something():
    for stage in STAGES:
        assert choices(stage), f"{stage} has no models at all"


def test_choices_are_ordered_cheapest_first():
    for stage in STAGES:
        tiers = [c.tier for c in choices(stage)]
        assert tiers == sorted(tiers)


def test_every_entry_records_a_licence_and_a_shipping_verdict():
    """The field that stops a licence problem being found at submission."""
    for stage in STAGES:
        for choice in choices(stage):
            assert choice.licence, f"{choice.key} has no licence recorded"
            assert isinstance(choice.shippable, bool)
            if not choice.shippable:
                assert choice.licence_note, (
                    f"{choice.key} cannot be shipped but does not say what to do"
                )


def test_the_models_actually_in_use_are_all_in_the_catalogue():
    """Otherwise the hardware page describes a system nobody is running."""
    for stage, key in in_use(DEFAULT).items():
        assert catalogue.find(key) is not None, f"{stage} runs {key}, absent from the catalogue"


def test_the_known_licence_blockers_are_reported_as_blockers():
    """YOLO11 and the InsightFace packs are the two that stop a sale.

    Pinned deliberately: if either is ever quietly marked shippable, that is a
    legal claim being made by a code change, and it should fail a test rather
    than a submission.
    """
    blocked = {c.key for c in unshippable()}
    assert "yolo11n" in blocked
    assert "buffalo_s" in blocked


def test_a_small_machine_is_not_offered_a_large_model():
    small = Machine(cpu_cores=4, ram_mb=4096)
    picked = recommend(small)
    for stage, choice in picked.items():
        assert choice.min_ram_mb <= small.ram_mb
        assert not choice.wants_gpu


def test_a_model_that_wants_a_real_gpu_is_refused_an_integrated_one():
    integrated = Accelerator("openvino", "GPU", usable=True)
    machine = Machine(cpu_cores=12, ram_mb=32768, accelerators=(integrated,))
    hungry = ModelChoice(
        key="test-big", stage="detect", title="Big", quality="",
        licence="MIT", shippable=True, min_ram_mb=8192, wants_gpu=True,
    )
    ok, why = fits(hungry, machine)
    assert ok is False
    assert "discrete GPU" in why


def test_an_unavailable_model_is_never_recommended():
    """Several tiers are listed but not yet wired up.

    Listing them is deliberate - it says where this is going - but suggesting
    one would send somebody to a model that cannot run.
    """
    machine = Machine(cpu_cores=16, ram_mb=65536,
                      accelerators=(Accelerator("onnxruntime", "CUDA",
                                                name="GeForce RTX 4090",
                                                vram_mb=24576, usable=True),))
    for choice in recommend(machine).values():
        assert choice.available, f"{choice.key} was recommended but is not available"


def test_recommendation_covers_every_stage_on_a_capable_machine():
    machine = Machine(cpu_cores=16, ram_mb=65536,
                      accelerators=(Accelerator("onnxruntime", "CUDA",
                                                name="GeForce RTX 4090",
                                                vram_mb=24576, usable=True),))
    assert set(recommend(machine)) == set(STAGES)


@pytest.mark.parametrize("stage", STAGES)
def test_keys_are_unique_within_a_stage(stage: str):
    keys = [c.key for c in choices(stage)]
    assert len(keys) == len(set(keys))
