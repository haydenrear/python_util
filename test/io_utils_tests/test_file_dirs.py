import os.path
from unittest import TestCase

from python_util.io_utils.file_dirs import iterate_files_in_directories, get_base_path_of_current_file, \
    get_test_work_dir, \
    delete_files_and_dirs_recursively, recursive_dir_iter
from python_util.io_utils.io import create_file


class Test(TestCase):
    def test_iterate_files_in_directories(self):
        next = False
        for i in iterate_files_in_directories(get_base_path_of_current_file(__file__)):
            next = True

        assert next

    def test_delete_recursively(self):
        test_work = get_test_work_dir(__file__)
        assert os.path.basename(test_work) == 'test_work'
        test_work = os.path.join(test_work, 'test_delete')
        if os.path.exists(test_work):
            delete_files_and_dirs_recursively(test_work)
        os.makedirs(test_work)
        for i in range(10):
            os.mkdir(os.path.join(test_work, str(i)))
            for j in range(10):
                to_create = os.path.join(test_work, str(i), f'{j}.txt')
                create_file(to_create)
                assert os.path.exists(to_create)

        delete_files_and_dirs_recursively(test_work)
        for i in range(10):
            for j in range(10):
                to_create = os.path.join(test_work, str(i), str(j))
                assert not os.path.exists(to_create)

    def test_read_recursively(self):
        count = 0
        for f in filter(lambda x: any([x.endswith(y) for y in ['.java', 'gradle.kts', '.py']]) and all([y not in x for y in ['phx']]), recursive_dir_iter('../../commit-diff-context')):
            if not os.path.isdir(f):
                print("opening " + f)
                with open(f, 'r') as to_read:
                    try:
                        for lin in to_read:
                            for c in lin:
                                count += 1
                    except UnicodeDecodeError as u:
                        print(u)
        print(count)