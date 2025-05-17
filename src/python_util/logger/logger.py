import logging
import os.path
import threading
import typing

from injector import synchronized

from python_util.logger.fluentd_logger import FluentDLogger, FluentDLoggerProperties, FluentDEvent
from python_util.logger.log_level import LogLevel, LogLevelFacade

lock = threading.RLock()


class LoggerFacade:
    ctx_values: typing.Optional[dict[str, str]]
    log_dir = os.environ.get('LOG_DIR') if 'LOG_DIR' in os.environ.keys() else None
    FLUENT_D_LOGGER: typing.Optional[FluentDLogger]

    @classmethod
    @synchronized(lock)
    def initialize_fluent_d(cls, fluent_d: FluentDLoggerProperties = FluentDLoggerProperties("http://localhost:8888")):
        if not hasattr(cls, "FLUENT_D_LOGGER"):
            if hasattr(cls, "ctx_values"):
                for k, v in cls.ctx_values.items():
                    if k not in fluent_d.ctx_values.keys():
                        fluent_d.ctx_values[k] = v

            cls.FLUENT_D_LOGGER = FluentDLogger(fluent_d)

    @classmethod
    @synchronized(lock)
    def register_ctx_value(cls, ctx_values: dict[str, str]):
        if cls.ctx_values is None:
            cls.ctx_values = ctx_values
        for k, v in ctx_values.items():
            if k not in cls.ctx_values.keys():
                cls.ctx_values[k] = v
            if k not in cls.FLUENT_D_LOGGER.ctx_values.keys():
                cls.FLUENT_D_LOGGER.ctx_values[k] = v

    @staticmethod
    def log_to_fluent_d(message: str, ctx_values: typing.Optional[dict[str, str]] = None,
                        label: str = "python", log_level: int = logging.INFO):
        if LogLevel.level == LogLevelFacade.Ctx:
            return
        if hasattr(LoggerFacade, 'FLUENT_D_LOGGER'):
            LoggerFacade.FLUENT_D_LOGGER.log_fluent_d(FluentDEvent(ctx_values, log_level, message, label))

    @staticmethod
    def debug(file_output, enable_debug: bool = False,
              ctx_values: typing.Optional[dict[str, str]] = None,
              label: str = "python"):
        level = LogLevel.level
        if level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(file_output)
            return
        if enable_debug:
            LogLevel.set_log_level(logging.DEBUG)
            assert LogLevel.is_debug_enabled()
        if LogLevel.is_debug_enabled() and LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/debug.log',
                      'a') as file:
                file.write(file_output)
                file.write('\n')
            print(f"Debug: {file_output}")
            logging.debug(file_output)
            LoggerFacade.log_to_fluent_d(file_output, ctx_values, label, logging.DEBUG)
        elif LogLevel.is_debug_enabled():
            print(f"Debug: {file_output}")
            logging.debug(file_output)
            LoggerFacade.log_to_fluent_d(file_output, ctx_values, label, logging.DEBUG)

    @staticmethod
    def debug_deferred(file_output: typing.Callable[[], str], enable_debug: bool = False,
                       ctx_values: typing.Optional[dict[str, str]] = None, label: str = "python"):
        if LogLevel.level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(file_output())
            return
        if enable_debug:
            LogLevel.set_log_level(logging.DEBUG)
            assert LogLevel.is_debug_enabled()
        if LogLevel.is_debug_enabled() and LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/debug.log',
                      'a') as file:
                deferred_value = file_output()
                file.write(deferred_value)
                file.write('\n')
            LoggerFacade.log_to_fluent_d(file_output(), ctx_values, label, logging.DEBUG)
        elif LogLevel.is_debug_enabled():
            deferred_value = file_output()
            print(f"Debug: {deferred_value}")
            logging.debug(deferred_value)
            LoggerFacade.log_to_fluent_d(file_output(), ctx_values, label, logging.DEBUG)

    @staticmethod
    def to_ctx(output,
               ctx_values: typing.Optional[dict[str, str]] = None,
               label: str = "python"):
        print(f'Context value: {output}')

    @staticmethod
    def info(output,
             ctx_values: typing.Optional[dict[str, str]] = None,
             label: str = "python"):
        if LogLevel.level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(output)
            return

        print(f"Info: {output}")
        logging.info(output)
        LoggerFacade.log_to_fluent_d(output, ctx_values, label, logging.INFO)
        if LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/info.log',
                      'a') as file:
                try:
                    file.write(output)
                    file.write('\n')
                except Exception as e:
                    print(f"Could not write in log: {e}")
    @staticmethod
    def raise_exc(output, exc_ty: typing.Union[typing.Type[Exception], Exception]):
        if LogLevel.level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(output)
            return
        LoggerFacade.error(output)
        if isinstance(exc_ty, Exception):
            raise exc_ty

        raise exc_ty(output)


    @staticmethod
    def error(output,
              ctx_values: typing.Optional[dict[str, str]] = None,
              label: str = "python"):
        if LogLevel.level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(output)
            return
        LoggerFacade.log_to_fluent_d(output, ctx_values, label, logging.ERROR)
        if LogLevel.is_error_write_enabled():
            LoggerFacade.write_to_log(output)
        print(f"Error: {output}")
        logging.error(output)

    @classmethod
    def write_to_log(cls, output, log = None):
        if log is None:
            if cls.log_dir is None:
                cls.log_dir = '/tmp/log.log'
            log = cls.log_dir

        if not os.path.exists(log):
            with open(log, 'w')  as file:
                pass
        with open(log, 'a') as file:
            file.write(output)
            file.write('\n')

    @staticmethod
    def warn(output,
             ctx_values: typing.Optional[dict[str, str]] = None,
             label: str = "python"):
        if LogLevel.level == LogLevelFacade.Ctx:
            LoggerFacade.write_to_log(output)
            return
        LoggerFacade.log_to_fluent_d(output, ctx_values, label, logging.WARN)
        print(f"Warning: {output}")
        logging.warning(output)
        LoggerFacade.info(output)
