"""最小的 WAV 封裝。

TTS.run() 產出 int16 單聲道（audio_postprocess 以 (audio * 32768).astype(np.int16)
結尾，TTS.py:1590），所以串流時直接送 raw PCM s16le 即可，不需要在請求路徑上
做任何編碼。只有「一次回傳完整音檔」的端點才需要補上 RIFF 標頭。
"""

from __future__ import annotations

import struct

BITS = 16
CHANNELS = 1


def header(sample_rate: int, data_bytes: int) -> bytes:
    """44 byte 的標準 RIFF/WAVE 標頭（PCM s16le 單聲道）。"""
    byte_rate = sample_rate * CHANNELS * BITS // 8
    block_align = CHANNELS * BITS // 8
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + data_bytes),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, CHANNELS, sample_rate, byte_rate, block_align, BITS),
            b"data",
            struct.pack("<I", data_bytes),
        )
    )


def wrap(sample_rate: int, pcm: bytes) -> bytes:
    return header(sample_rate, len(pcm)) + pcm
