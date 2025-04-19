__all__ = ["SmartCal"]

def _lazy_imports():
    global SmartCal, metrics
    
    from smartcal.smartcal.smartcal import SmartCal

_lazy_imports()
