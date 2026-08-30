"""Which models exist, what each costs, and whether it can be shipped.

Every stage of the pipeline can be run by more than one model, and the right
one is a property of the machine rather than of the software. A laptop with an
integrated GPU and a desktop with 12 GB of VRAM should not be handed the same
detector, and neither should be asked which they would prefer.

**Licensing is a field here, not a footnote.** Two of the models this project
started on cannot be shipped in a commercial product: Ultralytics YOLO11 is
AGPL-3.0, which extends its source obligations to any application distributed
with it, and the InsightFace `buffalo` packs are published for non-commercial
research. Both work perfectly well for someone running this on their own
footage, and both are blockers the moment it goes in a store - so the
catalogue records the difference rather than leaving it to be discovered at
submission.

The licence strings below are recorded from each project's own statements at
the time of writing. They are a starting point for a decision, not legal
advice, and anyone shipping this should confirm them against current terms.

**Bigger is not automatically better.** Nothing here promises a larger model
is faster or even more accurate on a given machine; the tiers are ordered by
cost, and `benchmark.py` measures what that cost actually is here. This module
only says what is *available* and what it *needs*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsv.hardware import Machine

# Stages a model can be chosen for. These are the keys used everywhere else.
STAGES: tuple[str, ...] = ("detect", "faces", "search", "captions", "audio")

STAGE_TITLES = {
    "detect": "Object detection",
    "faces": "Face recognition",
    "search": "Semantic search",
    "captions": "Descriptions",
    "audio": "Speech",
}


@dataclass(frozen=True)
class ModelChoice:
    """One model that can fill one stage."""

    key: str                  # stable id; also what gets recorded on vectors
    stage: str
    title: str
    # What picking it buys, in a sentence somebody can act on.
    quality: str

    licence: str
    # Whether it can go in a product that is sold or distributed. False here
    # is not "unusable" - it is "fine for yourself, not for shipping".
    shippable: bool
    licence_note: str = ""

    approx_mb: int = 0
    # Headroom the whole pipeline needs with this model loaded, not the file
    # size. Video decoding runs alongside it.
    min_ram_mb: int = 4096
    # True when a discrete GPU is what makes it practical rather than merely
    # faster. An integrated GPU does not count; measured, see backend.py.
    wants_gpu: bool = False
    # Ordering within a stage, cheapest first.
    tier: int = 0
    available: bool = True
    unavailable_reason: str = ""


# ---------------------------------------------------------------------------
# Detection
#
# The tier ladder that matters most: it runs on every sampled frame of every
# video, so it dominates import time.
#
# YOLO11 is what the project was built on and it is good, but AGPL-3.0 makes
# it the single biggest obstacle to shipping. YOLOX is Apache-2.0 and now
# runs: models/detect.py carries both families, and YOLOX needs no export
# step at all because its ONNX graphs are published directly.
#
# Measured on one real frame, same person, same box to within a few pixels:
# YOLO11n scored 0.918 and YOLOX-tiny 0.880. Close enough that the licence is
# the deciding factor rather than the accuracy.
# ---------------------------------------------------------------------------

_DETECT = (
    ModelChoice(
        key="yolo11n",
        stage="detect",
        title="YOLO11n",
        quality="the baseline. Fast enough for a night of footage on a laptop.",
        licence="AGPL-3.0",
        shippable=False,
        licence_note=(
            "AGPL extends to any application distributed with it. Fine on your "
            "own machine; for a product, either replace it or take an "
            "Ultralytics enterprise licence."
        ),
        approx_mb=11,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="yolo11s",
        stage="detect",
        title="YOLO11s",
        quality="finds smaller and more distant people than the baseline, at roughly twice the cost per frame.",
        licence="AGPL-3.0",
        shippable=False,
        licence_note="Same AGPL obligation as YOLO11n.",
        approx_mb=36,
        min_ram_mb=6144,
        tier=1,
    ),
    ModelChoice(
        key="yolo11m",
        stage="detect",
        title="YOLO11m",
        quality="noticeably better on crowded or low-light scenes; wants a real GPU to stay practical.",
        licence="AGPL-3.0",
        shippable=False,
        licence_note="Same AGPL obligation as YOLO11n.",
        approx_mb=77,
        min_ram_mb=8192,
        wants_gpu=True,
        tier=2,
    ),
    ModelChoice(
        key="yolox-tiny",
        stage="detect",
        title="YOLOX-tiny",
        quality="the shippable baseline: measured at 0.880 against YOLO11n's 0.918 on the same person, under a licence that allows selling the result.",
        licence="Apache-2.0",
        shippable=True,
        approx_mb=20,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="yolox-s",
        stage="detect",
        title="YOLOX-s",
        quality="the shippable step up, roughly where YOLO11s sits.",
        licence="Apache-2.0",
        shippable=True,
        approx_mb=36,
        min_ram_mb=6144,
        tier=1,
    ),
)

# ---------------------------------------------------------------------------
# Faces
#
# Resolved, with a caveat worth keeping in view.
#
# OpenCV's zoo publishes YuNet (MIT) and SFace (Apache-2.0), and OpenCV is
# already a dependency here, so the permissive stack installs nothing new.
# Detection is a wash rather than an upgrade - over 98 person crops SCRFD
# found a face in 20.4% and YuNet in 18.4%, median 17-21 pixels either way -
# which is the right result: the swap costs no accuracy and removes the last
# research-only licence.
#
# The caveat is that SFace's *recognition* could not be validated on the
# footage available. At the face sizes this footage contains, same-person and
# different-person pairs score identically. That is a property of 18-pixel
# faces rather than of the model, and no face model fixes it - see
# models/face_opencv.py for the numbers.
#
# The InsightFace entries stay because they work and are legitimate for
# personal use; they are simply marked as what they are.
# ---------------------------------------------------------------------------

_FACES = (
    ModelChoice(
        key="yunet-sface",
        stage="faces",
        title="YuNet + SFace",
        quality="lets a person be named once and recognised elsewhere, under licences that allow selling the result.",
        licence="MIT (YuNet) + Apache-2.0 (SFace)",
        shippable=True,
        licence_note="",
        approx_mb=40,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="buffalo_s",
        stage="faces",
        title="SCRFD det_500m + ArcFace w600k_mbf",
        quality="lets a person be named once and recognised in other sightings.",
        licence="InsightFace: non-commercial research use",
        shippable=False,
        licence_note=(
            "Published for research. No permissive replacement is in this "
            "catalogue yet - face recognition weights are usually bound by the "
            "terms of the dataset they were trained on."
        ),
        approx_mb=16,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="buffalo_l",
        stage="faces",
        title="SCRFD det_10g + ArcFace w600k_r50",
        quality="recognises faces at greater distance and sharper angles; about four times the cost.",
        licence="InsightFace: non-commercial research use",
        shippable=False,
        licence_note="Same research-only terms as buffalo_s.",
        approx_mb=326,
        min_ram_mb=8192,
        tier=1,
        available=False,
        unavailable_reason="not yet wired into the face pipeline",
    ),
)

# ---------------------------------------------------------------------------
# Semantic search
#
# Changing this one is the expensive change: every stored vector was produced
# by a particular model and is meaningless under another, so switching means
# re-embedding the whole index. The `model` column on the embedding tables is
# what makes that safe rather than silently wrong.
# ---------------------------------------------------------------------------

_SEARCH = (
    ModelChoice(
        key="clip-vit-b-32",
        stage="search",
        title="CLIP ViT-B/32",
        quality="searching by description. The baseline, and the only one the index is built for today.",
        licence="MIT",
        shippable=True,
        approx_mb=605,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="clip-vit-l-14",
        stage="search",
        title="CLIP ViT-L/14",
        quality="markedly better at fine detail - colours, held objects, clothing - and several times the cost per image.",
        licence="MIT",
        shippable=True,
        approx_mb=1700,
        min_ram_mb=12288,
        wants_gpu=True,
        tier=1,
        available=False,
        unavailable_reason="needs the re-embedding pass before it can be selected",
    ),
)

# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

_CAPTIONS = (
    ModelChoice(
        key="florence2-base-ft",
        stage="captions",
        title="Florence-2-base-ft",
        quality="describes what a person is doing, so actions and held objects become searchable.",
        licence="MIT",
        shippable=True,
        approx_mb=275,
        min_ram_mb=6144,
        tier=0,
    ),
    ModelChoice(
        key="florence2-large-ft",
        stage="captions",
        title="Florence-2-large-ft",
        quality="names what base only gestures at - measured on the same crop, base gave \"a mannequin standing on a chair\" where this gave \"a woman standing on a set of stairs, a pink hat, holding a metal pole\". About three times the cost per image, and this is already the slowest stage.",
        licence="MIT",
        shippable=True,
        approx_mb=1540,
        min_ram_mb=12288,
        wants_gpu=True,
        tier=1,
    ),
)

# ---------------------------------------------------------------------------
# Speech
#
# Measured on the baseline machine (i5-1235U, int8, CPU) against 500 seconds
# of audio with voice-activity detection *off*, so the whole file is really
# decoded - the honest wall-to-wall-speech cost rather than the speed of
# skipping silence:
#
#     tiny    25.0x realtime    2.4 minutes per hour of speech
#     base    17.8x realtime    3.4 minutes per hour
#     small    8.0x realtime    7.5 minutes per hour
#
# With VAD on and real footage, which is mostly silence, base ran at 112x.
#
# Cores buy almost nothing: 2 threads gave 19.4x and all 12 gave 20.6x, a 6%
# gain for six times the cores. Whisper's decoder is autoregressive and this
# workload is not core-bound, so a bigger CPU is the wrong thing to spend on.
# A discrete GPU is where the headroom is - see audio.pick_device.
# ---------------------------------------------------------------------------

_AUDIO = (
    ModelChoice(
        key="whisper-tiny",
        stage="audio",
        title="Whisper tiny",
        quality="the fastest, and the one that drops or invents words on distant or noisy speech.",
        licence="MIT",
        shippable=True,
        approx_mb=78,
        min_ram_mb=4096,
        tier=0,
    ),
    ModelChoice(
        key="whisper-base",
        stage="audio",
        title="Whisper base",
        quality="the default: the smallest worth trusting on a room recording, at 3.4 minutes per hour of speech.",
        licence="MIT",
        shippable=True,
        approx_mb=148,
        min_ram_mb=4096,
        tier=1,
    ),
    ModelChoice(
        key="whisper-medium",
        stage="audio",
        title="Whisper medium",
        quality=(
            "the usual answer for a language base cannot hold, and measured here "
            "it is not: on Bengali it kept nothing across three runs, gave two "
            "different answers to the same file, and ran at 0.18x realtime - "
            "190s for a 35s clip, against base's 2s."
        ),
        licence="MIT",
        shippable=True,
        approx_mb=1530,
        min_ram_mb=16384,
        tier=3,
    ),
    ModelChoice(
        key="whisper-small",
        stage="audio",
        title="Whisper small",
        quality="better on accents and distance in English, but measured here it failed on Bengali - wrong script, and nothing above the confidence floor - while running 29x slower than base rather than the 2x its size suggests.",
        licence="MIT",
        shippable=True,
        approx_mb=486,
        min_ram_mb=8192,
        tier=2,
    ),
)

CATALOGUE: dict[str, tuple[ModelChoice, ...]] = {
    "detect": _DETECT,
    "faces": _FACES,
    "search": _SEARCH,
    "captions": _CAPTIONS,
    "audio": _AUDIO,
}


def choices(stage: str, available_only: bool = False) -> tuple[ModelChoice, ...]:
    entries = CATALOGUE.get(stage, ())
    if available_only:
        entries = tuple(c for c in entries if c.available)
    return tuple(sorted(entries, key=lambda c: (c.tier, c.approx_mb)))


def find(key: str) -> ModelChoice | None:
    for entries in CATALOGUE.values():
        for choice in entries:
            if choice.key == key:
                return choice
    return None


def unshippable() -> tuple[ModelChoice, ...]:
    """Everything that blocks distributing this as a product.

    Surfaced as a list rather than a warning per model, because the question
    it answers - "can I sell this yet" - is about the set, not the parts.
    """
    return tuple(
        choice
        for stage in STAGES
        for choice in choices(stage)
        if choice.available and not choice.shippable
    )


def fits(choice: ModelChoice, machine: Machine) -> tuple[bool, str]:
    """Whether this machine can run it, and why not when it cannot."""
    if not choice.available:
        return False, choice.unavailable_reason
    if machine.ram_mb and machine.ram_mb < choice.min_ram_mb:
        need = choice.min_ram_mb / 1024
        have = machine.ram_mb / 1024
        return False, f"needs about {need:.0f} GB of RAM, this machine has {have:.0f} GB"
    if choice.wants_gpu and not machine.has_discrete_gpu:
        return False, "needs a discrete GPU to run at a sensible speed"
    return True, ""


def in_use(cfg) -> dict[str, str]:
    """The key of whatever each stage is configured to run right now.

    Separate from `recommend`, and the distinction matters on screen: one is
    what this machine could carry, the other is what it is actually doing.
    Presenting a suggestion as the current state is how somebody concludes
    they have already upgraded.
    """
    return {
        "detect": cfg.detect.model_file.replace(".onnx", ""),
        "faces": cfg.face.name,
        "search": cfg.clip.name,
        "captions": cfg.caption.name,
        "audio": cfg.audio.model_dir,
    }


def recommend(machine: Machine) -> dict[str, ModelChoice]:
    """The largest model per stage this machine can actually carry.

    Deliberately the *largest that fits* rather than the best benchmarked:
    fitting is a property of the hardware and can be decided here, while
    "better" is a measurement that belongs to benchmark.py and to the user's
    own footage. Nothing in this function claims one model finds more than
    another.
    """
    picked: dict[str, ModelChoice] = {}
    for stage in STAGES:
        for choice in choices(stage, available_only=True):
            if fits(choice, machine)[0]:
                picked[stage] = choice
    return picked
