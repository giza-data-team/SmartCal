__all__ = ["ECE", "MCE", "ConfECE", "calculate_brier_score", "calculate_calibration_curve"]

def _lazy_imports():
    global ECE, MCE, ConfECE, calculate_brier_score, calculate_calibration_curve
    
    from smartcal.metrics.ece import ECE
    from smartcal.metrics.mce import MCE
    from smartcal.metrics.conf_ece import ConfECE
    from smartcal.metrics.brier_score import calculate_brier_score
    from smartcal.metrics.calibration_curve import calculate_calibration_curve

_lazy_imports()
