import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np


class VideoDecodeBenchmarkTests(unittest.TestCase):
    def test_real_cpu_decode_sparse_and_reuse_match_dense_pixels(self):
        try:
            from torchcodec.decoders import VideoDecoder
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"Working torchcodec is unavailable: {exc}")
        spec = importlib.util.spec_from_file_location("decode_bench", Path(__file__).resolve().parents[1] / "scripts/benchmark_video_decode.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            video = Path(tmp) / "fixture.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                stream.codec_context.gop_size = 12
                for index in range(48):
                    rgb = np.empty((48, 64, 3), dtype=np.uint8)
                    rgb[:, :, 0] = index * 5
                    rgb[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :] * 3
                    rgb[:, :, 2] = np.arange(48, dtype=np.uint8)[:, None] * 4
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            result = module.run_benchmark(video, windows=3, warmup=1)
        self.assertTrue(result["all_selected_frames_identical"])
        self.assertEqual([row["requested_frames"] for row in result["results"]], [33, 9, 33, 9])
        self.assertEqual([row["decoder_constructions_including_warmup"] for row in result["results"]], [4, 4, 1, 1])
        for row in result["results"]:
            self.assertEqual(row["selected_frame_hash_matches"], 3)


if __name__ == "__main__":
    unittest.main()
