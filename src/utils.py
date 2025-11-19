import matplotlib.pyplot as plt

class Location:
    def __init__(self, x: int = -1, y: int = -1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State:
    def __init__(self, time: int, location: Location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self) -> int:
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state) -> bool:
        return self.location == state.location

    def return_location_lst(self) -> list[int, int]:
        return [self.location.x, self.location.y]

    def __str__(self) -> str:
        return str((self.time, self.location.x, self.location.y))
