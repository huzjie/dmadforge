import unittest

from dmadforge.audiovideo.codec import AudioCodec
from dmadforge.audiovideo.fusion import JointLatentFusion


class TestAudioVideo(unittest.TestCase):
    def test_codec_roundtrip_shape(self):
        codec = AudioCodec(audio_dim=4, latent_dim=4, seed="c")
        z = codec.encode([[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(len(z[0]), 4)
        back = codec.decode(z)
        self.assertEqual(len(back[0]), 4)

    def test_fusion_consistency_identical(self):
        f = JointLatentFusion(latent_dim=4)
        v = [[0.1, 0.2, 0.3, 0.4]]
        self.assertAlmostEqual(f.consistency(v, v), 1.0, places=5)

    def test_fusion_consistency_opposite(self):
        f = JointLatentFusion(latent_dim=4)
        v = [[1.0, 1.0, 1.0, 1.0]]
        a = [[-1.0, -1.0, -1.0, -1.0]]
        c = f.consistency(v, a)
        self.assertLess(c, 0.5)


if __name__ == "__main__":
    unittest.main()
