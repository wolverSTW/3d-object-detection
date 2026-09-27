import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.evaluation.kitti_eval import KittiEvaluator
from src.losses.detection_loss import DetectionLoss
from src.losses.geometry_loss import GeometryLoss
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config


def load_kitti_pipeline_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kitti_pipeline.py"
    spec = importlib.util.spec_from_file_location("kitti_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPipelineBasics(unittest.TestCase):
    def test_load_yaml_config(self):
        config = load_yaml_config('configs/experiments/baseline.yaml')
        self.assertIn('training', config)
        self.assertEqual(config['training']['epochs'], 100)

    def test_detection_loss_returns_positive_value(self):
        criterion = DetectionLoss(lambda_box=1.0, lambda_cls=0.5, lambda_3d=1.5)
        outputs = {
            'box_pred': [1.0, 2.0, 3.0],
            'cls_pred': [0.8, 0.2],
            'dim_pred': [1.0, 1.0, 1.0],
        }
        targets = {
            'box_target': [0.5, 1.5, 2.5],
            'cls_target': [1.0, 0.0],
            'dim_target': [0.8, 0.9, 1.1],
        }
        loss = criterion(outputs, targets)
        self.assertGreater(loss, 0.0)

    def test_geometry_loss_returns_positive_value(self):
        criterion = GeometryLoss(lambda_geo=0.8)
        outputs = {'depth_pred': [10.0, 12.0], 'proj_pred': [11.0, 13.0]}
        targets = {'depth_target': [8.0, 10.0], 'proj_target': [10.0, 12.0]}
        loss = criterion(outputs, targets)
        self.assertGreater(loss, 0.0)

    def test_trainer_epoch_returns_loss(self):
        trainer = Trainer(model=None, optimizer=None, criterion=None, save_dir='experiments/test')
        dataloader = [{'image': None, 'annotations': [], 'calib': {}} for _ in range(2)]
        loss = trainer.train_epoch(dataloader)
        self.assertGreaterEqual(loss, 0.0)

    def test_evaluator_returns_metrics(self):
        evaluator = KittiEvaluator()
        predictions = [{'distance': 10.0, 'class': 'Car'}, {'distance': 12.0, 'class': 'Car'}]
        ground_truth = [{'distance': 9.0, 'class': 'Car'}, {'distance': 13.0, 'class': 'Car'}]
        metrics = evaluator.evaluate(predictions, ground_truth)
        self.assertIn('MAE_distance', metrics)
        self.assertIn('RMSE_distance', metrics)
        self.assertGreaterEqual(metrics['MAE_distance'], 0.0)
        self.assertGreaterEqual(metrics['RMSE_distance'], 0.0)

    def test_trainer_fit_returns_history(self):
        trainer = Trainer(model=None, optimizer=None, criterion=None, save_dir='experiments/test_fit')
        dataloader = [{'image': None, 'annotations': [], 'calib': {}} for _ in range(2)]
        history = trainer.fit(dataloader, epochs=2)
        self.assertIn('loss', history)
        self.assertGreaterEqual(len(history['loss']), 2)

    def test_evaluator_supports_dataset_level_summary(self):
        evaluator = KittiEvaluator()
        results = evaluator.evaluate_dataset(
            [{'distance': 10.0, 'AP3D_easy': 0.70}, {'distance': 12.0, 'AP3D_easy': 0.68}],
            [{'distance': 9.0}, {'distance': 13.0}],
        )
        self.assertIn('MAE_distance', results)
        self.assertIn('RMSE_distance', results)
        self.assertIn('AP3D_easy', results)

    def test_download_replaces_invalid_zip(self):
        module = load_kitti_pipeline_module()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("sample.txt", "hello")
        valid_bytes = buffer.getvalue()

        class DummyResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=1024 * 1024):
                yield self.payload

        def fake_get(url, stream=True, timeout=60):
            return DummyResponse(valid_bytes)

        original_get = module.requests.get
        module.requests.get = fake_get
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                data_root = Path(tempdir) / "KITTI"
                raw_root = data_root / "raw"
                raw_root.mkdir(parents=True)
                bad_archive = raw_root / "data_object_image_2.zip"
                bad_archive.write_text("not a zip", encoding="utf-8")

                result = module.download_kitti_dataset(data_root)

                self.assertEqual(result["download_required"], True)
                self.assertTrue(module.archive_is_valid(bad_archive))
                self.assertIn(str(bad_archive), result["downloaded"])
        finally:
            module.requests.get = original_get


if __name__ == '__main__':
    unittest.main()
