import unittest

from src.evaluation.kitti_eval import KittiEvaluator
from src.losses.detection_loss import DetectionLoss
from src.losses.geometry_loss import GeometryLoss
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config


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


if __name__ == '__main__':
    unittest.main()
