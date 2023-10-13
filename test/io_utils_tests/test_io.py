import os
from unittest import TestCase

from python_util.io_utils.io import write_bytes_to_disk


class TestIo(TestCase):
    def test_write_bytes_to_disk(self):
        resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.bin")

        write_bytes_to_disk("hello".encode("utf-8"), resources)

        with open(resources, "rb") as file:
            next = file.readline()
            next = next.decode('utf-8')
            assert next == "hello"

        assert os.path.exists(resources)

        if os.path.exists(resources):
            os.remove(resources)
