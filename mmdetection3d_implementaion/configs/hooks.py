from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class MetricsDebugHook(Hook):
    """Logs the raw metrics dict after each validation epoch to ensure key names."""

    priority = "LOW"

    def after_val_epoch(self, runner, metrics=None):
        runner.logger.info(f"[metrics_debug] {metrics}")
