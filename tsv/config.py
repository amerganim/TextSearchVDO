"""Tunables for the Phase 0 ingest pipeline.

Every threshold here is a guess until it has been run against real footage.
They are grouped and named so they can be swept from a script later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TierAConfig:
    """Packet-size prefilter: finds candidate windows without decoding."""

    bin_seconds: float = 1.0
    # Candidates are picked by robust z-score (median/MAD) of mean P-frame
    # size. Tier B re-checks everything this stage passes, so a false positive
    # only costs decode time whereas a miss is unrecoverable - hence a
    # deliberately generous threshold.
    z_threshold: float = 3.0
    # Floor on the MAD, as a fraction of baseline. Without it a very stable
    # stream has MAD ~= 0 and every bin scores as infinitely anomalous.
    min_rel_mad: float = 0.03
    # A bin must also be this much above baseline, so trivial fluctuation on a
    # near-static stream cannot pass on z-score alone.
    min_ratio: float = 1.08
    # The baseline is measured over a rolling window rather than the whole
    # file. A camera's noise floor changes hugely across a recording - IR
    # cutover at dusk is the obvious one, but rain and auto-exposure do it too
    # - and a single global baseline lets the noisier half hide its own
    # activity behind a threshold set by the quieter half.
    baseline_window_seconds: float = 120.0
    # Bins are smoothed over this many neighbours before thresholding. The
    # motion premium on a P-frame is itself GOP-position dependent - frames
    # just after a keyframe stay cheap even while something is moving - so a
    # single burst arrives as an alternating series that breaks contiguity at
    # one-second resolution. Tier A only has to decide *whether* to decode a
    # stretch, so trading that resolution away costs nothing.
    smooth_bins: int = 3
    # Quantile of the bin distribution taken as the idle floor. Not the
    # median: motion only pushes bins upward, so on a clip where someone is on
    # screen half the time the median sits inside the activity and hides it.
    idle_quantile: float = 0.30
    # Candidate windows are padded before being handed to Tier B, so segment
    # edges are found by the accurate stage rather than the cheap one.
    pad_seconds: float = 2.0
    # Above this fraction of keyframes the stream is effectively all-intra
    # (MJPEG, or a camera with GOP=1) and packet sizes carry no motion signal.
    all_intra_keyframe_ratio: float = 0.9


@dataclass(frozen=True)
class TierBConfig:
    """Decode-and-subtract stage: accurate scoring on candidate windows."""

    sample_fps: float = 3.0
    width: int = 320
    mog2_history: int = 120
    mog2_var_threshold: float = 24.0
    # Morphological opening kernel, kills single-pixel sensor noise.
    open_kernel: int = 3
    # Blobs smaller than this fraction of the frame are ignored entirely.
    min_blob_area_frac: float = 0.0008
    # A frame scores above this fraction of changed pixels to count as active.
    active_area_frac: float = 0.0025
    # Mean-luminance jump that indicates an IR day/night switch rather than
    # motion. On a 0-255 scale.
    ir_flip_luma_delta: float = 18.0
    # Frames to discard after an IR flip while MOG2 relearns the background.
    ir_flip_cooldown_frames: int = 8
    # Frames used to prime MOG2 before its output is trusted.
    warmup_frames: int = 10


@dataclass(frozen=True)
class SegmentConfig:
    """Turning a per-sample activity score into human-meaningful segments."""

    # Hysteresis: open a segment at `open_frac`, keep it alive until activity
    # drops below `close_frac`. Prevents flicker on borderline motion.
    open_frac: float = 0.0035
    close_frac: float = 0.0015
    # Bridge gaps shorter than this: someone pausing mid-frame is one event.
    merge_gap_seconds: float = 3.0
    # Discard anything shorter than this: usually a bird or a noise spike.
    min_duration_seconds: float = 1.0
    # Context around each segment so playback starts before the action.
    pre_roll_seconds: float = 1.5
    post_roll_seconds: float = 1.5


@dataclass(frozen=True)
class DetectConfig:
    """Phase 1: what to run the detector on, and how hard."""

    model_file: str = "yolo11n.onnx"
    # Which lineage the file belongs to, deciding channel order, value range
    # and padding - all of which fail silently when wrong. None reads it from
    # the filename, which is right for every model this project installs; set
    # it explicitly for a graph named something else.
    family: str | None = None
    # A starting point only. These exports are static and carry their own
    # size - YOLOX-tiny is 416 - so the detector reads it from the graph and
    # falls back to this.
    input_size: int = 640
    # Frames per second sampled for detection. Higher costs linearly and buys
    # tracking stability; 4 keeps a walking person's boxes overlapping enough
    # to associate while staying affordable on a CPU.
    detect_fps: float = 4.0
    # Frames are decoded at this width before letterboxing. Decoding 1080p to
    # feed a 640px network is wasted bandwidth, but going too small loses the
    # distant figures that matter most on a wide camera.
    decode_width: int = 960
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    # None means the CCTV subset in models.detect; an explicit tuple overrides.
    classes: tuple[str, ...] | None = None
    # "runtime:device" to pin a backend, e.g. "onnxruntime:CPU". None selects.
    force_backend: str | None = None
    crop_width: int = 160


@dataclass(frozen=True)
class FaceConfig:
    """Phase 2: who a person is."""

    # Which weights these are, recorded on every vector they produce. Two
    # face models can share a dimension and share nothing else, so a stored
    # vector is meaningless without knowing what made it - see `model` on the
    # embedding tables.
    name: str = "buffalo_s"
    detector_file: str = "det_500m.onnx"
    embedder_file: str = "w600k_mbf.onnx"
    # "insightface" (SCRFD + ArcFace, research licence) or "opencv" (YuNet +
    # SFace, MIT and Apache-2.0). The second is the one that can be shipped.
    # Cosine thresholds differ between them, so this travels with `name`.
    stack: str = "insightface"
    det_size: int = 640
    conf_threshold: float = 0.5
    # A face smaller than this carries almost no identity information; a
    # confident twelve-pixel face is still worthless.
    min_face_px: int = 24
    # Faces are embedded from the best few crops of each tracklet rather than
    # every sampled frame. Running the face stack per frame would cost more
    # than the detector that found the person in the first place, and the
    # extra views add little once the best ones are in hand.
    max_faces_per_tracklet: int = 5
    # These graphs are small enough that an integrated GPU loses to the CPU;
    # see SMALL_MODEL_PREFERENCE in models/backend.py.
    force_backend: str | None = None


@dataclass(frozen=True)
class AudioConfig:
    """Phase 5: what the recording heard.

    On by default where the model is present, for the same reason captioning
    is: a transcript nobody asked for is a search that quietly finds nothing.
    Most cameras record no audio at all, in which case this costs one demux
    per file and stops.
    """

    enabled: bool = True
    model_dir: str = "whisper-base"
    # "auto" takes a discrete GPU where there is one and falls back to the
    # CPU, which is the difference between a workstation transcribing at CPU
    # speed and using what it has. "cpu" pins it.
    device: str = "auto"
    # None follows the device: float16 on a GPU, int8 on a CPU. Set it to
    # override both. On speech this noisy the quality difference against
    # float32 is small and the speed difference is not.
    compute_type: str | None = None
    # None lets the model detect it per file. Setting it is markedly faster
    # and stops a noisy recording being decided as the wrong language.
    language: str | None = None
    threads: int = 0

    name: str = "whisper-base"


@dataclass(frozen=True)
class ClipConfig:
    """Phase 3: semantic search."""

    # As with faces, recorded on every vector. This one is sharper: CLIP
    # ViT-B/32 and ViT-B/16 are both 512-dimensional and entirely
    # incompatible, so mixing them raises no error at all - it silently
    # returns nonsense similarity. The name is what stops that.
    name: str = "clip-vit-b-32"
    image_file: str = "clip_image.onnx"
    text_file: str = "clip_text.onnx"
    size: int = 224
    # "pad" keeps the whole subject; "center" reproduces CLIP's own
    # preprocessing at the cost of a standing person's head and feet.
    crop_mode: str = "pad"
    force_backend: str | None = None
    # Semantic matches below this cosine score are discarded, so a query for
    # something that is not there answers "nothing found" instead of handing
    # back the closest three frames in the library.
    #
    # The library deliberately has no default floor - a caller that can see
    # the scores should choose its own. An app cannot: its whole promise is an
    # answer or an honest nothing, and ranking everything forever reads as
    # broken. So the number lives here, is shown to the user as a percentage
    # on every result, and is one line to change.
    #
    # 0.20 comes from measurement, but on few samples: a matching query scored
    # 0.22-0.27 and an unrelated one 0.15. Expect to move it once real footage
    # and real questions have been through it.
    min_similarity: float = 0.20


@dataclass(frozen=True)
class CaptionConfig:
    """Phase 4: describing what a person is doing.

    On whenever the model is present, because a description that has not been
    written yet is a search that quietly fails: "carrying a bag" matches
    nothing, and the app cannot tell the reader that the answer might be there
    but undescribed. Somebody who has to remember to press a button before
    searching will not.

    It is genuinely the slow part - about six seconds an image on a CPU, and
    the cost is per *image*, so a night of footage with a few hundred person
    tracklets is the better part of an hour. Two things make that bearable
    rather than a stall: it runs last, so motion, objects and search are all
    live long before it starts, and the progress bar weights it honestly (see
    STAGE_SHARES_WITH_CAPTIONS) instead of appearing to hang at 95%.
    """

    enabled: bool = True
    model_dir: str = "florence2"
    # Length is nearly free once the image is encoded, so ask for the most
    # detailed description available: more words means more to search.
    task: str = "more_detailed"
    max_tokens: int = 64
    # Only people are captioned. A description of a parked car earns nothing
    # that the object label did not already say.
    labels: tuple[str, ...] = ("person",)
    # Extra room around the detector's box. What someone is holding is usually
    # just outside it.
    context: float = 0.35
    # Below this, a crop carries no describable detail and the model invents.
    min_crop_px: int = 96
    force_backend: str | None = None


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    # Models are shared between indexes rather than belonging to one, so this
    # can point somewhere common; it defaults to <data_dir>/models.
    model_dir_override: Path | None = None
    tier_a: TierAConfig = field(default_factory=TierAConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tier_b: TierBConfig = field(default_factory=TierBConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    thumb_width: int = 320
    thumb_quality: int = 80

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def thumb_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def crop_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def model_dir(self) -> Path:
        return self.model_dir_override or (self.data_dir / "models")

    @property
    def detect_model_path(self) -> Path:
        return self.model_dir / self.detect.model_file

    @property
    def face_detector_path(self) -> Path:
        return self.model_dir / self.face.detector_file

    @property
    def face_embedder_path(self) -> Path:
        return self.model_dir / self.face.embedder_file

    @property
    def clip_image_path(self) -> Path:
        return self.model_dir / self.clip.image_file

    @property
    def clip_text_path(self) -> Path:
        return self.model_dir / self.clip.text_file

    @property
    def has_clip_models(self) -> bool:
        return self.clip_image_path.is_file() and self.clip_text_path.is_file()

    @property
    def caption_model_dir(self) -> Path:
        return self.model_dir / self.caption.model_dir

    @property
    def has_audio_model(self) -> bool:
        directory = self.audio_model_dir
        return directory.is_dir() and (directory / "model.bin").is_file()

    @property
    def audio_model_dir(self) -> Path:
        return self.model_dir / self.audio.model_dir

    @property
    def has_caption_model(self) -> bool:
        needed = ("vision_encoder", "embed_tokens", "encoder_model", "decoder_model_merged")
        d = self.caption_model_dir
        return all((d / f"{n}.onnx").is_file() for n in needed) and (d / "prompts.json").is_file()

    @property
    def has_face_models(self) -> bool:
        return self.face_detector_path.is_file() and self.face_embedder_path.is_file()

    @property
    def opencv_face(self) -> "Config":
        """The same configuration with the permissive face stack selected."""
        import dataclasses as _dc

        return _dc.replace(
            self,
            face=_dc.replace(
                self.face,
                name="yunet-sface",
                stack="opencv",
                detector_file="face_detection_yunet_2023mar.onnx",
                embedder_file="face_recognition_sface_2021dec.onnx",
                # SFace's own threshold, on its own scale. See
                # models/face_opencv.SFACE_SAME_PERSON.
                conf_threshold=0.6,
            ),
        )


DEFAULT = Config()
