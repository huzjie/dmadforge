"""32kHz audio codec: map raw audio frames into the shared latent width.

MiniMax H3 emits 32kHz stereo; we model audio as a sequence of frames each
projected to `latent_dim`, so the same denoiser handles audio and video.
"""
from ..core.nn import Linear
from ..core.tensor import matmul


class AudioCodec:
    def __init__(self, audio_dim=4, latent_dim=4, seed=""):
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.enc = Linear(audio_dim, latent_dim, seed + "aenc")
        self.dec = Linear(latent_dim, audio_dim, seed + "adec")

    def encode(self, audio_frames):
        return self.enc(audio_frames)

    def decode(self, z):
        return self.dec(z)
