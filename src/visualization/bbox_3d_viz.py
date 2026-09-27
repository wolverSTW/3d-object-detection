class BBox3DVisualizer:
    """Simple 3D bounding-box visualization helper."""

    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir

    def render(self, sample_id, box_3d):
        return {
            'sample_id': sample_id,
            'box_3d': box_3d,
            'status': 'rendered',
        }
