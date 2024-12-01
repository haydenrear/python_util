import dataclasses
import typing

import requests


@dataclasses.dataclass(init=True)
class FluentDEvent:
    ctx: typing.Optional[dict[str, str]]
    log_level: int
    message: str
    label: str


@dataclasses.dataclass(init=True)
class FluentDLoggerProperties:
    url: str
    ctx_values: typing.Optional[dict[str, str]] = None


class FluentDLogger:
    def __init__(self, fluent_d_logger: FluentDLoggerProperties = FluentDLoggerProperties("http://localhost:8888")):
        self.s = lambda x: requests.post(url=f"{fluent_d_logger.url}", json=x,
                                         headers={"Content-Type": "application/json"})
        self.ctx_values = fluent_d_logger.ctx_values if fluent_d_logger.ctx_values is not None else {}

    def log_fluent_d(self, fluent_d_event: FluentDEvent):
        r = self.s({
            **(fluent_d_event.ctx if fluent_d_event.ctx is not None else {}),
            **{k: v for k, v in self.ctx_values.items() if k not in fluent_d_event.ctx.keys()},
            "message": fluent_d_event.message
        })
        if not str(r.status_code).startswith("2"):
            print(f"Error sending fluentD log: {r.content}")
