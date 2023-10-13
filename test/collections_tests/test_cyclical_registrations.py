import asyncio
import unittest

from python_util.collections.cyclic_registrations import CyclicRegistrations


class C:
    c = 0
    l = 0


class TestCyclicalRegistrations(unittest.IsolatedAsyncioTestCase):

    def update_c(self):
        C.c += 1

    def update_l(self):
        C.c += 1

    async def go(self, cyclical):
        await cyclical.arrive(1)
        await cyclical.await_current()

    async def test_cyclical(self):
        cyclical = CyclicRegistrations({1, 2, 3}, self.update_c)

        out = self.go(cyclical)

        await cyclical.arrive(2)
        await cyclical.arrive(3)
        await out

        assert len(cyclical.current) == 0
        assert C.c == 1


    async def test_cyclical_clear(self):
        cyclical = CyclicRegistrations({1, 2, 3}, self.update_c)

        await cyclical.arrive(2)
        await cyclical.arrive(3)
        await cyclical.arrive(1)

        cyclical.clear()

        assert len(cyclical.current) == 0
        assert C.c == 1


