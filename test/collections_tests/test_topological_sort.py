import unittest

from python_util.collections.topological_sort import HasDependencies, topological_sort, T


class DependencyObject(HasDependencies):

    def __init__(self, name, depends_on=None):
        self.name = name
        self.depends_on = depends_on or []

    def __repr__(self):
        return f"Object: {self.name}, Depends on: {self.depends_on}"

    def get_dependencies(self):
        return self.depends_on

    def self_id(self) -> T:
        return 'hello'

# Example objects with dependencies
obj1 = DependencyObject("Obj1")
obj2 = DependencyObject("Obj2", depends_on=["Obj1"])
obj3 = DependencyObject("Obj3", depends_on=["Obj2"])
obj4 = DependencyObject("Obj4", depends_on=["Obj3"])
obj5 = DependencyObject("Obj5", depends_on=["Obj4"])
obj6 = DependencyObject("Obj6", depends_on=["Obj5"])
obj7 = DependencyObject("Obj7", depends_on=["Obj6"])
# obj9 = DependencyObject("Obj9", depends_on=["Obj8"])

objects = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]






