from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from subtitle_creator import cli, engine


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "movie.mp4"
        self.video.touch()
        self.output = self.video.with_suffix(".srt")
        self.stdout = redirect_stdout(io.StringIO())
        self.stderr = redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_discovery_supports_case_insensitive_extensions_and_recursion(self):
        (self.root / "second.MKV").touch()
        (self.root / "ignore.txt").touch()
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "third.MP4").touch()
        self.assertEqual(len(cli.discover_videos(self.root, False)), 2)
        self.assertEqual(len(cli.discover_videos(self.root, True)), 3)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            cli.discover_videos(self.root / "missing", False)

    def test_existing_srt_does_not_load_model_or_probe_cuda(self):
        self.output.write_text("original")
        with patch.object(cli, "cuda_available") as probe, patch.object(engine, "create_model") as model:
            self.assertEqual(cli.main([str(self.root)]), 0)
        probe.assert_not_called()
        model.assert_not_called()
        self.assertEqual(self.output.read_text(), "original")

    def test_timestamp_rounding_carries_to_next_hour_and_preserves_unicode(self):
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(start=3599.9996, end=3601.1234, text="  Živjo   svet!  ")]),
            SimpleNamespace(language="sl", language_probability=0.99, duration=3602),
        )
        self.assertEqual(engine.write_transcription(model, [], self.output, "sl"), 1)
        self.assertEqual(self.output.read_text(), "1\n01:00:00,000 --> 01:00:01,123\nŽivjo svet!\n\n")
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "sl")

    def test_ffmpeg_fallback_uses_first_audio_track_and_cleans_temp(self):
        def extract(command, **kwargs):
            with wave.open(command[-1], "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x40")
            return SimpleNamespace(returncode=0)

        with patch("faster_whisper.audio.decode_audio", side_effect=ValueError("codec")), \
                patch.object(engine.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                patch.object(engine.subprocess, "run", side_effect=extract) as run:
            self.assertEqual(engine.load_audio(self.video).tolist(), [0.5])
        command = run.call_args.args[0]
        self.assertIn("0:a:0", command)
        self.assertFalse(Path(command[-1]).exists())

    def test_missing_ffmpeg_has_actionable_error(self):
        with patch("faster_whisper.audio.decode_audio", side_effect=ValueError("codec")), \
                patch.object(engine.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "dnf install ffmpeg-free"):
                engine.load_audio(self.video)

    def test_cpu_failure_preserves_previous_srt_and_cleans_partial_output(self):
        self.output.write_text("previous subtitles")

        def fail(model, audio, output, language):
            output.write_text("partial")
            raise RuntimeError("inference failed")

        with patch.object(engine, "load_audio", return_value=[]), \
                patch.object(engine, "create_model"), \
                patch.object(engine, "write_transcription", side_effect=fail):
            self.assertEqual(cli.main([str(self.video), "--force", "--device", "cpu"]), 1)
        self.assertEqual(self.output.read_text(), "previous subtitles")
        self.assertEqual(list(self.root.glob(".subtitle-*")), [])

    def test_cuda_failure_retries_cpu_and_disables_cuda_for_remaining_files(self):
        (self.root / "second.mkv").touch()

        def transcribe(model, audio, output, language):
            output.write_text("complete")
            return 1

        with patch.object(cli, "run_cuda", return_value=False) as gpu, \
                patch.object(engine, "load_audio", return_value=[]), \
                patch.object(engine, "create_model") as model, \
                patch.object(engine, "write_transcription", side_effect=transcribe):
            self.assertEqual(cli.main([str(self.root), "--device", "cuda"]), 0)
        self.assertEqual(gpu.call_count, 1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(model.call_args.args[1], "cpu")
        self.assertEqual(self.output.read_text(), "complete")
        self.assertEqual((self.root / "second.srt").read_text(), "complete")

    def test_batch_continues_after_a_bad_video(self):
        (self.root / "second.mkv").touch()

        def transcribe(model, audio, output, language):
            output.write_text("complete")
            return 1

        with patch.object(engine, "load_audio", side_effect=[ValueError("bad video"), []]), \
                patch.object(engine, "create_model"), \
                patch.object(engine, "write_transcription", side_effect=transcribe):
            self.assertEqual(cli.main([str(self.root), "--device", "cpu"]), 1)
        self.assertFalse(self.output.exists())
        self.assertTrue((self.root / "second.srt").exists())

    def test_no_clobber_publication_and_force(self):
        source = self.root / "partial"
        source.write_text("new")
        self.output.write_text("existing")
        self.assertFalse(cli.publish(source, self.output, False))
        self.assertEqual(self.output.read_text(), "existing")
        self.assertTrue(cli.publish(source, self.output, True))
        self.assertEqual(self.output.read_text(), "new")

    def test_same_stem_collision_is_rejected_before_transcription(self):
        self.video.with_suffix(".mkv").touch()
        with self.assertRaises(SystemExit) as error:
            cli.main([str(self.root), "--force"])
        self.assertEqual(error.exception.code, 2)

    def test_ctrl_c_cleans_partial_file(self):
        def interrupt(model, audio, output, language):
            output.write_text("partial")
            raise KeyboardInterrupt()

        with patch.object(engine, "load_audio", return_value=[]), \
                patch.object(engine, "create_model"), \
                patch.object(engine, "write_transcription", side_effect=interrupt):
            self.assertEqual(cli.main([str(self.video), "--device", "cpu"]), 130)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".subtitle-*")), [])

    def test_native_gpu_crash_is_a_recoverable_failure(self):
        args = cli.build_parser().parse_args([str(self.video), "--offline", "--language", "sl"])
        with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=-6)) as run:
            self.assertFalse(cli.run_cuda(self.video, self.output, args))
        self.assertIn("--offline", run.call_args.args[0])
        self.assertIn("sl", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
