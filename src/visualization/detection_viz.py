class DetectionVisualizer:
    """Simple visualization helper for object predictions and distances."""

    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir

    def render(self, sample_id, prediction):
        return {
            'sample_id': sample_id,
            'prediction': prediction,
            'status': 'rendered',
        }
