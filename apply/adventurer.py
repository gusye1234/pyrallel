class Adventurer:
    """Go through a function to log every operation
    """
    ID = 0
    Arithmetic_support = [int, float]
    def __init__(self,
                 name="",
                 before=[]):
        self.id = Adventurer.ID
        self.name = name if name else f"NoName-{self.id}"
        Adventurer.ID += 1
        self.path = before

    def __add__(self, number):
        assert type(number) in Adventurer.Arithmetic_support
        return Adventurer(
            name=self.name,
            before=self.path + [OperationLog('ADD', number)])

    def __radd__(self, number):
        return self.__add__(number)

    def __iadd__(self, number):
        return self.__add__(number)

    def __neg__(self):
        return Adventurer(name=self.name,
                          before=self.path + [OperationLog('NEG')])

    def __repr__(self):
        name = self.name + ':\n'
        for op in self.path:
            name = name + str(op)
        return name

class OperationLog:
    """Data structure to log function
    """
    def __init__(self, f_name, args=None):
        self._function = f_name
        self._args = args or ""

    def __repr__(self):
        return f"-> {self._function} {self._args}\n"

    def __str__(self):
        return self.__repr__()