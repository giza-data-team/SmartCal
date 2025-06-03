__all__ = ["SmartCal"]

def _lazy_imports():
    global SmartCal
    
    from smartcal.smartcal.smartcal import SmartCal

_lazy_imports()
