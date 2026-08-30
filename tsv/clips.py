"""A few seconds around a moment, instead of the whole recording.

Tapping a search result used to hand the browser the entire file and ask it to
seek. On a desktop that is invisible; on a phone over WiFi it is the
difference between watching something and waiting for it. One of the videos
here is 124 MB for half an hour, and the answer somebody wants from it is ten
seconds long.

Cutting those ten seconds out costs 0.05 seconds and produces 0.68 MB - about
a hundred and eighty times less to send - because nothing is re-encoded. The
packets are copied straight across into a new container, so this is bounded by
disk and demuxing rather than by any codec.

**It starts at a keyframe, which is earlier than asked for.** A stream copy
cannot begin mid-GOP: the first frame has to be one that stands alone, or
everything until the next keyframe decodes as garbage. So the clip opens a
little before the moment - which is what somebody wants anyway, since the
interesting part is usually the approach.

**The codec is whatever the camera recorded, unless asked otherwise.** Copying
preserves it, so a phone's HEVC stays HEVC - which is right by default, since
phones both record and play HEVC. Desktop Firefox does not, and neither do
some older Androids, so a client that knows it cannot decode HEVC asks for
H.264 and gets it.

That fallback is a real re-encode and costs accordingly. Measured on a twelve
second clip of 960x1080 HEVC:

    copy (no re-encode)             0.01s   0.82 MB
    libx264 veryfast crf20          1.00s   3.98 MB
    libx264 veryfast crf23          0.89s   2.85 MB
    libx264 veryfast crf28          0.84s   1.69 MB    <- the fallback
    libx264 veryfast crf34          1.80s   0.93 MB
    h264_qsv (Intel iGPU)           will not open on this machine

crf 28 because the point of the exercise is sending less over WiFi: it costs
the same time as crf 23 for forty percent less data, and this is a clip for
working out what happened, not an archival copy - the footage has already
been through the camera's own compression once. The hardware encoder is
*listed* by ffmpeg here and refuses to open, which is the same "listed is not
usable" rule the backend layer lives by, so nothing here depends on one.

Asking for web-safe output on a recording that is already H.264 costs
nothing: it falls through to the copy path, measured at 0.04s against 0.03s.
"""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

# How much to include before the moment, when the keyframe allows it. Enough
# to see somebody arrive rather than appear.
LEAD_IN = 2.0

# Clips are meant to be small. A request for ten minutes is either a mistake
# or somebody using this as a download endpoint; either way, the whole file is
# already available from /api/media.
MAX_SECONDS = 60.0


# What a browser is guaranteed to play. Anything else is a gamble on the
# viewer's decoder.
WEB_SAFE = frozenset({"h264", "vp8", "vp9", "av1"})

# The re-encode settings, chosen by measurement rather than by taste. See the
# table at the top of this file.
H264_OPTIONS = {"crf": "28", "preset": "veryfast"}


def _clean_rate(rate) -> "Fraction":
    """A frame rate an encoder will accept.

    Real recordings report absurd ones. This laptop's phone footage says its
    average rate is 26996000/1799749 - which is 15.0 fps to six decimal
    places, and which becomes the encoder's time base denominator, where
    libx264 rejects it with a bare "Invalid argument" naming nothing.

    A rate that is already expressible sensibly is kept exactly - 30000/1001
    is how NTSC is spelled and rounding it would be worse than leaving it.
    Anything else is rounded to a thousandth of a frame, which turns that
    fraction into 15/1.
    """
    from fractions import Fraction

    try:
        exact = Fraction(rate)
        value = float(exact)
    except (TypeError, ValueError, ZeroDivisionError):
        return Fraction(30, 1)

    if not (1.0 <= value <= 240.0):
        return Fraction(30, 1)
    if exact.denominator <= 1001:
        return exact
    return Fraction(round(value, 3)).limit_denominator(1000)


def source_codec(path: Path) -> str:
    """The video codec in a file, lowercased, or "" if it cannot be read."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return ""
            return (container.streams.video[0].codec_context.name or "").lower()
    except Exception:      # noqa: BLE001 - an unreadable file has no codec
        return ""


def cut(
    path: Path,
    at: float,
    seconds: float = 10.0,
    lead_in: float = LEAD_IN,
    web_safe: bool = False,
) -> bytes:
    """An MP4 of the moments around `at`, copied rather than re-encoded.

    `web_safe` asks for something any browser can play. Where the recording is
    already H.264 that changes nothing and costs nothing; where it is HEVC the
    video is re-encoded, which is the only expensive thing in this module.

    Raises FileNotFoundError if the recording has moved, and ValueError if
    there is nothing there to cut - a timestamp past the end, most often.
    """
    import av

    if not path.is_file():
        raise FileNotFoundError(path)

    if web_safe and source_codec(path) not in WEB_SAFE:
        return _cut_transcoded(path, at, seconds, lead_in)

    seconds = max(1.0, min(float(seconds), MAX_SECONDS))
    start = max(0.0, float(at) - lead_in)

    buffer = io.BytesIO()
    with av.open(str(path)) as source:
        if not source.streams.video:
            raise ValueError("no video stream")
        video = source.streams.video[0]
        audio = source.streams.audio[0] if source.streams.audio else None

        with av.open(buffer, mode="w", format="mp4") as target:
            out_video = target.add_stream_from_template(video)
            out_audio = target.add_stream_from_template(audio) if audio else None

            # Seeking lands on the keyframe at or before this point, which is
            # why the clip can begin earlier than requested.
            try:
                source.seek(int(start / video.time_base), stream=video)
            except (av.FFmpegError, OverflowError, ValueError):
                raise ValueError("cannot seek to that moment") from None

            streams = [video] + ([audio] if audio else [])
            written = 0
            origin: float | None = None

            for packet in source.demux(streams):
                # A flush packet at end of stream has no dts and must not be
                # muxed; it is the demuxer telling us it is done.
                if packet.dts is None:
                    continue
                when = (
                    float(packet.pts * packet.time_base)
                    if packet.pts is not None else 0.0
                )
                if origin is None:
                    origin = when
                if when > start + seconds:
                    break

                target_stream = (
                    out_video if packet.stream.type == "video" else out_audio
                )
                if target_stream is None:
                    continue

                # Move the clip's timestamps back to zero. Copied packets keep
                # the presentation times they had in the source, so without
                # this a twelve second clip taken from three minutes in
                # declares itself three minutes long: the player draws a
                # scrubber across the whole recording, starts near the end of
                # it, and the twelve seconds that actually exist are a sliver.
                shift = int(origin / packet.time_base)
                packet.pts = None if packet.pts is None else packet.pts - shift
                packet.dts = packet.dts - shift
                packet.stream = target_stream
                target.mux(packet)
                written += 1

    if not written:
        raise ValueError("nothing to cut at that moment")
    return buffer.getvalue()


def _cut_transcoded(
    path: Path, at: float, seconds: float, lead_in: float
) -> bytes:
    """The same cut, with the video re-encoded to H.264.

    Only the video: audio is copied, because AAC is as universal as H.264 and
    re-encoding it would be work for nothing.
    """
    import av

    seconds = max(1.0, min(float(seconds), MAX_SECONDS))
    start = max(0.0, float(at) - lead_in)

    buffer = io.BytesIO()
    with av.open(str(path)) as source:
        if not source.streams.video:
            raise ValueError("no video stream")
        video = source.streams.video[0]
        audio = source.streams.audio[0] if source.streams.audio else None

        with av.open(buffer, mode="w", format="mp4") as target:
            out_video = target.add_stream(
                "libx264",
                rate=_clean_rate(video.average_rate or video.guessed_rate),
                options=dict(H264_OPTIONS),
            )
            out_video.width = video.codec_context.width
            out_video.height = video.codec_context.height
            out_video.pix_fmt = "yuv420p"
            out_audio = target.add_stream_from_template(audio) if audio else None

            try:
                source.seek(int(start / video.time_base), stream=video)
            except (av.FFmpegError, OverflowError, ValueError):
                raise ValueError("cannot seek to that moment") from None

            streams = [video] + ([audio] if audio else [])
            origin: float = 0.0
            seen_origin = False
            written = 0
            # Frames are numbered at a constant rate rather than carrying
            # their source timing across.
            #
            # Timing is where this goes wrong, and both obvious approaches
            # fail. Passing the source pts through unchanged is rejected -
            # they are in the source's time base and start wherever the seek
            # landed. Clearing them and letting the encoder number frames is
            # rejected too, with "non-strictly-monotonic PTS". Computing them
            # from real presentation time still fails on a variable frame rate
            # recording, because two frames closer together than one tick of
            # the encoder's time base land on the same dts and the muxer
            # refuses: "non monotonically increasing dts ... 6000 >= 6000".
            #
            # Counting is the one thing that cannot collide. The cost is that
            # a variable rate source becomes constant rate, and audio drifts
            # by the difference between the real and nominal rates - which for
            # the recording this was built against is 15.000009 against 15, so
            # a tenth of a millisecond over a twelve second clip.
            frame_tb = Fraction(1, 1) / _clean_rate(
                video.average_rate or video.guessed_rate
            )
            index = 0

            for packet in source.demux(streams):
                if packet.dts is None:
                    continue
                when = (
                    float(packet.pts * packet.time_base)
                    if packet.pts is not None else 0.0
                )
                if not seen_origin:
                    origin = when
                    seen_origin = True
                if when > start + seconds:
                    break

                if packet.stream.type == "audio":
                    if out_audio is None:
                        continue
                    shift = int(origin / packet.time_base)
                    packet.pts = None if packet.pts is None else packet.pts - shift
                    packet.dts = packet.dts - shift
                    packet.stream = out_audio
                    target.mux(packet)
                    continue

                for frame in packet.decode():
                    frame.pts = index
                    frame.time_base = frame_tb
                    index += 1
                    for encoded in out_video.encode(frame):
                        target.mux(encoded)
                        written += 1

            for encoded in out_video.encode():
                target.mux(encoded)
                written += 1

    if not written:
        raise ValueError("nothing to cut at that moment")
    return buffer.getvalue()
