from pandapower.shortcircuit.run import runsc

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
logger.warning("WARNING: pandapower shortcircuit module is in beta stadium, proceed with caution!")