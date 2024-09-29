import unittest
from unittest import mock

from python_util.logger.logger import LoggerFacade


class FluentDLoggerTest(unittest.TestCase):
    def test_fluent_d(self):
        LoggerFacade.initialize_fluent_d()
        LoggerFacade.FLUENT_D_LOGGER.ctx_values["whatever"] = "hey"

        class Res:
            status_code = 200

        magic_mock = unittest.mock.MagicMock(return_value=Res())
        LoggerFacade.FLUENT_D_LOGGER.s = magic_mock
        LoggerFacade.info("Hello!", {"hello": "goodbye"})
        c = magic_mock.mock_calls[0]

        assert c[1][0]['hello'] == 'goodbye'
        assert c[1][0]['whatever'] == 'hey'
        assert c[1][0]['message'] == 'Hello!'


if __name__ == '__main__':
    unittest.main()
