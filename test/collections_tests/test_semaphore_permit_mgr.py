import asyncio
import unittest

from python_util.collections.cyclic_registrations import CyclicRegistrations, UnderlyingSet, ListCyclicRegistrations


class TestSemaphorePermitManager(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)
        self.test_state = []

    def test_semaphore_permit_mgr_test(self):
        asyncio.get_event_loop().run_until_complete(self.set_permit_mgr(lambda: CyclicRegistrations({'one.txt', 'two.txt'})))

    def test_semaphore_permit_mgr_test_aset(self):
        asyncio.get_event_loop().run_until_complete(self.set_permit_mgr(lambda: ListCyclicRegistrations(['one.txt', 'two.txt'])))

    async def set_permit_mgr(self, ty_cb):
        permit_mgr = ty_cb()
        await permit_mgr.register('three')
        await permit_mgr.arrive('one.txt')
        await permit_mgr.arrive('two.txt')
        out = asyncio.get_event_loop().create_task(self.try_wait(permit_mgr))
        await asyncio.sleep(1)
        assert len(self.test_state) == 0
        await permit_mgr.arrive('three')
        await out
        assert len(self.test_state) == 1

    async def try_wait(self, permit_mgr: CyclicRegistrations):
        await permit_mgr.await_current()
        self.test_state.append('three')





if __name__ == '__main__':
    unittest.main()
