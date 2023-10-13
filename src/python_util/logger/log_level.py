import logging


class LogLevel:
    level = None


    @classmethod
    def is_debug_enabled(cls):
        return cls.level and cls.level == logging.DEBUG

    @classmethod
    def set_log_level(cls, name):
        cls.level = name

    @classmethod
    def get_log_level(cls):
        return cls.level
