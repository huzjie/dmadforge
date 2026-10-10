"""Fuse video and audio latents into one joint sequence.

The key idea (from H3's native audio-video generation): rather than two
separate denoisers, concatenate video latents and audio latents along the
sequence axis and denoise them together, so audio-visual synchrony is learned
implicitly by the shared attention.
"""
from ..core.tensor import matmul


class JointLatentFusion:
    def __init__(self, latent_dim=4):
        self.latent_dim = latent_dim

    def fuse(self, video_z, audio_z):
        return video_z + audio_z

    def split(self, z, n_video):
        return z[:n_video], z[n_video:]

    def consistency(self, video_z, audio_z):
        """Audio-visual consistency: 1 - normalized mean distance.

        1.0 means video and audio latents are identical (perfectly in sync,
        which happens when skill -> 1 and both come from the shared gold).
        0.0 means they are completely unrelated (independent noise).
        """
        n = max(1, len(video_z))
        c = len(video_z[0]) if video_z else self.latent_dim
        dist = 0.0
        vn = 0.0
        an = 0.0
        for i in range(n):
            for j in range(c):
                diff = video_z[i][j] - audio_z[i][j]
                dist += diff * diff
                vn += video_z[i][j] * video_z[i][j]
                an += audio_z[i][j] * audio_z[i][j]
        denom = vn + an
        if denom <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, 1.0 - dist / denom))
