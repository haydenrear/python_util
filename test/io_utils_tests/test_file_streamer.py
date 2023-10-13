import os
import unittest

from python_util.io_utils.file_dirs import get_dir
from python_util.io_utils.file_streamer import FileStreamer


class TestFileStreamer(unittest.TestCase):
    def test_load_chunk(self):
        file_ = os.path.join(get_dir(__file__, 'work'), 'test_file.txt')
        with open(file_, 'w') as file:
            for i in range(1000):
                file.writelines(f'{i}\n')

        file_streamer = FileStreamer(file_, 'utf-8')
        assert str(file_streamer[0]).replace('\n', '') == '0'
        assert str(next(file_streamer)).replace('\n', '') == '0'
        assert str(next(file_streamer)).replace('\n', '') == '1'
        assert str(file_streamer[100]).replace('\n', '') == '100'
        assert str(file_streamer[400]).replace('\n', '') == '400'
        assert str(next(file_streamer)).replace('\n', '') == '2'
        assert str(file_streamer[401]).replace('\n', '') == '401'
