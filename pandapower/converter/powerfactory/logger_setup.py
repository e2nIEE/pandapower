from logging import StreamHandler, Formatter

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging


class AppHandler(StreamHandler):
    def __init__(self, app, freeze_app_between_messages=False):
        super().__init__()
        self.app = app
        self.PrintPlain = app.PrintPlain
        self.PrintInfo = app.PrintInfo
        self.PrintWarn = app.PrintWarn
        self.PrintError = app.PrintError
        self.name = 'PowerFactory Converter'
        formatter = Formatter('%(message)s')
        self.setFormatter(formatter)
        self.freeze_app_between_messages = freeze_app_between_messages

    def emit(self, record):
        if self.freeze_app_between_messages:
            self.app.SetGuiUpdateEnabled(1)

        msg = self.format(record)
        level = record.levelname
        if level == "DEBUG":
            self.PrintPlain(msg)
        elif level == "INFO":
            self.PrintInfo(msg)
        elif level == "WARNING":
            self.PrintWarn(msg)
        elif level == "ERROR":
            self.PrintError(msg)
        elif level == 'CRITICAL':
            self.PrintError(msg)
        else:
            self.PrintPlain(msg)

        if self.freeze_app_between_messages:
            self.app.SetGuiUpdateEnabled(0)


def setup_logger(app, name, level):
    logger = logging.getLogger(name)

    app_handler = AppHandler(app)
    logger.addHandler(app_handler)

    set_PF_level(logger, app_handler, level)

    logger.info('initialized logger %s' % logger.name)
    return logger, app_handler


def set_PF_level(logger, app_handler, level):
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        app_handler.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
        app_handler.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
        app_handler.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
        app_handler.setLevel(logging.ERROR)
