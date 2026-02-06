"""
This module is a container for Vector class.
"""

from __future__ import annotations
import struct

class Vector:
    """
    Represents a mathematical vector.
    """
    _precision: int = 5
    def __init__(self, init_list: list) -> None:
        if not init_list:
            raise ValueError
        self._vector: list = init_list
        self._n_dimentions: int = len(self._vector)

    def __repr__(self) -> str:
        return f"Vector({self._vector})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self._vector == other._vector

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._vector)

    def norm(self) -> float:
        """
        Calculates the Euclidean norm (L^2 norm) of the vector which is 
        a measure of the  "length" or "magnitude" of the vector.

        :param self: 
        :return: norm of the vector
        :rtype: float
        """
        return sum(v*v for v in self._vector) ** 2

    def __lt__(self, other: Vector) -> bool:
        return self.norm() < other.norm()

    def __gt__(self, other: Vector) -> bool:
        return not self.__lt__(other)

    def __le__(self, other: Vector) -> bool:
        return self.norm() <= other.norm()

    def __ge__(self, other: Vector) -> bool:
        return not self.__le__(other)

    def __str__(self) -> str:
        return f"Vector {self._vector} has {self._n_dimentions} {"dimention" if self._n_dimentions == 1 else "dimentions"}"

    def __bool__(self) -> bool:
        return any(self._vector)

    def __int__(self) -> int:
        return NotImplemented

    def __float__(self) -> float:
        return NotImplemented

    def __bytes__(self) -> bytes:
        return struct.pack("I", self._n_dimentions)+"".join(struct.pack("d", v) for v in self._vector).encode()

    def __complex__(self) ->complex:
        return NotImplemented

    def __format__(self, spec: str) -> str:
        if not spec:
            return str(self._vector)
        return "["+", ".join(format(v, spec) for v in self._vector)+"]"

    class _PrecisionContext:
        def __init__(self, vector_object: Vector, precision: int) -> None:
            self._vector_object: Vector = vector_object
            self._precision: int = precision
            self._old_precision: int = None

        def __enter__(self) -> Vector:
            self._old_precision = self._precision
            self._vector_object._precision = self._precision
            return self._vector_object

        def __exit__(self, exc_type, exc, traceback) -> None:
            self._vector_object._precision = self._old_precision

    def precision(self, precision: int) -> _PrecisionContext:
        """
        Docstring for precision
        
        :param self: This instance of the class Vector.
        :param precision: Precision for each number stored inside Vector.
        :type precision: int
        :return: Object that handles context managing used when "with" syntax.
        :rtype: _PrecisionContext
        """
        return self._PrecisionContext(self, precision)

