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

**The codec is whatever the camera recorded.** Copying preserves it, so a
phone's HEVC stays HEVC. That is the right default: phones play HEVC natively,
and re-encoding to be safe would turn a 0.05 second operation into several.
Where it matters, the full file has exactly the same constraint - this makes
nothing worse.
"""

from __future__ import annotations

import io
from pathlib import Path

# How much to include before the moment, when the keyframe allows it. Enough
# to see somebody arrive rather than appear.
LEAD_IN = 2.0

# Clips are meant to be small. A request for ten minutes is either a mistake
# or somebody using this as a download endpoint; either way, the whole file is
# already available from /api/media.
MAX_SECONDS = 60.0


def cut(
    path: Path,
    at: float,
    seconds: float = 10.0,
    lead_in: float = LEAD_IN,
) -> bytes:
    """An MP4 of the moments around `at`, copied rather than re-encoded.

    Raises FileNotFoundError if the recording has moved, and ValueError if
    there is nothing there to cut - a timestamp past the end, most often.
    """
    import av

    if not path.is_file():
        raise FileNotFoundError(path)

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
