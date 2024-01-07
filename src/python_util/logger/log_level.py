import logging


class LogLevel:
    level = None
    do_write = False


    @classmethod
    def is_debug_enabled(cls):
        return cls.level and cls.level == logging.DEBUG

    @classmethod
    def enable_do_write(cls):
        cls.do_write = True

    @classmethod
    def is_write_enabled(cls):
        return cls.do_write

    @classmethod
    def set_log_level(cls, name):
        cls.level = name

    @classmethod
    def get_log_level(cls):
        return cls.level
