"""
This module is a container for Vector class.
"""

from __future__ import annotations
import struct
from collections.abc import Iterator
from typing import Sequence, Any, TYPE_CHECKING
import asyncio
import math
if TYPE_CHECKING:
    from .matrix import Matrix

# pylint: disable=import-outside-toplevel

class Vector:
    """
    Represents a mathematical vector.
    """
    class Precision:
        """
        Inner class that represents python descriptor.
        """
        def __init__(self, default: int | None = None):
            self.default: int | None = default
            self.name = None

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, instance, owner) -> int:
            if instance is None:
                return self
            return instance.__dict__.get(self.name, self.default or 0)

        def __set__(self, instance, value) -> None:
            if not 0 <= value <= 9:
                raise ValueError("precision must be in [0, 9]")
            instance.__dict__[self.name] = value

        def __delete__(self, instance) -> None:
            instance.__dict__[self.name] = self.default or 0

    _precision: Precision = Precision()

    def __init__(self, init_list: list[float], n_dimentions: int) -> None:
        if n_dimentions <= 0:
            raise ValueError
        if len(init_list) > n_dimentions:
            raise ValueError("Initial list longer than dimention")
        self._vector = list(init_list) + [0] * (n_dimentions - len(init_list))
        self._n_dimentions: int = n_dimentions

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
        return math.sqrt(sum(v*v for v in self._vector))

    def __lt__(self, other: Vector) -> bool:
        if math.isclose(self.norm(), other.norm(), rel_tol=1e-14):
            return False
        return self.norm() < other.norm()

    def __gt__(self, other: Vector) -> bool:
        if math.isclose(self.norm(), other.norm(), rel_tol=1e-14):
            return False
        return not self.__lt__(other)

    def __le__(self, other: Vector) -> bool:
        if math.isclose(self.norm(), other.norm(), rel_tol=1e-14, abs_tol=1e-14):
            return True
        return self.norm() <= other.norm()

    def __ge__(self, other: Vector) -> bool:
        if math.isclose(self.norm(), other.norm(), rel_tol=1e-14, abs_tol=1e-14):
            return True
        return self.norm() >= other.norm()

    def __str__(self) -> str:
        return (
            f"Vector {self._vector} has {self._n_dimentions} "
            f"{'dimention' if self._n_dimentions == 1 else 'dimentions'}"
        )

    def __bool__(self) -> bool:
        return any(self._vector)

    def __int__(self) -> int:
        if self._n_dimentions == 1:
            return int(self._vector[0])
        raise TypeError("Cannot convert multidimentional Vector to int")

    def __float__(self) -> float:
        if self._n_dimentions == 1:
            return float(self._vector[0])
        raise TypeError("Cannot convert multidimentional Vector to float")

    def __bytes__(self) -> bytes:
        return (
            struct.pack("I", self._n_dimentions)+
            b"".join(struct.pack("d", v) for v in self._vector)
        )

    def __complex__(self) -> complex:
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
            self._old_precision = self._vector_object._precision
            self._vector_object._precision = self._precision
            return self._vector_object

        def __exit__(self, exc_type, exc, traceback) -> bool:
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

    def __len__(self) -> int:
        return self._n_dimentions

    class _VectorIterator(Iterator[float]):
        def __init__(self, vector: Sequence[float]) -> None:
            self.current_index = 0
            self.vector = vector

        def __iter__(self) -> Vector._VectorIterator:
            return self

        def __next__(self) -> float:
            current = self.current_index
            self.current_index += 1
            if current >= len(self.vector):
                raise StopIteration
            return self.vector[current]

        def __length_hint__(self):
            return len(self.vector) - self.current_index

    def __iter__(self) -> _VectorIterator:
        """
        This could be also implemented using build-in function that creates iterator:
            def __iter__(self):
                return iter(self._values)
        or using generator that creates iterator using yield:
            def __iter__(self):
                for x in self._values:
                    yield x


        :param self: Description
        :return: Description
        :rtype: VectorIterator
        """
        return self._VectorIterator(self._vector)

    def __getitem__(self, index: int | slice) -> float | Vector:
        result = self._vector[index]
        if isinstance(index, slice):
            return Vector(result, len(result))
        return result

    def __setitem__(self, index: int | slice, value: float | list[float]) -> None:
        if isinstance(index, slice):
            if len(value) != len(self._vector[index]):
                raise ValueError("Vector dimention change")
        self._vector[index] = value

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("Vector size cannot be changed")

    def __contains__(self, value: float) -> bool:
        return value in self._vector

    def __reversed__(self) -> Iterator[float]:
        return reversed(self._vector)

    def __call__(self, other: Vector) -> float:
        """
        Implementation of __call__ magic method for Vector class

        :param self: 
        :param other: other vector of the same dimention
        :type other: Vector
        :return: Dot product of two vectors
        :rtype: float
        """
        if self._n_dimentions != other._n_dimentions:
            raise ValueError
        return sum(a*b for a, b in zip(self._vector, other._vector))

    def __add__(self, other: Vector | int | float) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions != other._n_dimentions:
                raise ValueError("Vector dimention mismatch")
            return Vector([a+b for a, b in zip(self._vector, other._vector)], self._n_dimentions)
        if isinstance(other, (int, float)):
            return Vector([v + other for v in self._vector], self._n_dimentions)
        raise ValueError

    def __sub__(self, other: Vector | int | float) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions != other._n_dimentions:
                raise ValueError("Vector dimention mismatch")
            return Vector([a-b for a, b in zip(self._vector, other._vector)], self._n_dimentions)
        if isinstance(other, (int, float)):
            return Vector([v - other for v in self._vector], self._n_dimentions)
        raise ValueError

    def __mul__(self, other: Vector | int | float | Matrix) -> Vector:
        """
        For multiplication with Vector the result is cross product 
        of two vectors which exists only in 3D.

        :param self: Description
        :param other: right side operand
        :type other: Vector | int | float | Matrix
        :return: Result of multiplying vector with other variables of other type
        :rtype: Vector
        """
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions and self._n_dimentions == 3:
                a1, a2, a3 = self._vector
                b1, b2, b3 = other._vector
                return Vector([a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1], self._n_dimentions)
            raise ValueError
        if isinstance(other, (int, float)):
            return Vector([v*other for v in self._vector], self._n_dimentions)
        # To avoid circular imports between vector.py and matrix.py
        from .matrix import Matrix
        if isinstance(other, Matrix):
            vector_list = []
            for i in range(other._n_columns):
                vector_list[i] = sum(
                    a*b for a, b in zip(self._vector, other.column[i]))
            return Vector(vector_list, len(vector_list))
        return NotImplemented

    def __rmul__(self, other: Vector | int | float | Matrix) -> Vector:
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __truediv__(self, divisor: int | float) -> Vector:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Vector([v/divisor for v in self._vector], self._n_dimentions)

    def __mod__(self, divisor: int | float) -> Vector:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Vector([v % divisor for v in self._vector], self._n_dimentions)

    def __floordiv__(self, divisor: int | float) -> Vector:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Vector([v//divisor for v in self._vector], self._n_dimentions)

    def __pow__(self, power: int) -> Vector:
        return Vector([v**power for v in self._vector], self._n_dimentions)

    def __and__(self, other: Vector | int) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions:
                return Vector(
                    [a & b for a, b in zip(self._vector, other._vector)], self._n_dimentions
                )
            raise ValueError
        if isinstance(other, int):
            return Vector([v & other for v in self._vector], self._n_dimentions)
        return NotImplemented

    def __rand__(self, other: int) -> Vector:
        return self.__and__(other)

    def __or__(self, other: Vector | int) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions:
                return Vector(
                    [a | b for a, b in zip(self._vector, other._vector)], self._n_dimentions
                )
            raise ValueError
        if isinstance(other, int):
            return Vector([v | other for v in self._vector], self._n_dimentions)
        return NotImplemented

    def __ror__(self, other: int) -> Vector:
        return self.__or__(other)

    def __xor__(self, other: Vector | int) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions:
                return Vector(
                    [a ^ b for a, b in zip(self._vector, other._vector)], self._n_dimentions
                )
            raise ValueError
        if isinstance(other, int):
            return Vector([v ^ other for v in self._vector], self._n_dimentions)
        return NotImplemented

    def __rxor__(self, other: int) -> Vector:
        return self.__xor__(other)

    def __rshift__(self, other: int) -> Vector:
        if isinstance(other, int):
            return Vector([v >> other for v in self._vector], self._n_dimentions)
        return NotImplemented

    def __lshift__(self, other: int) -> Vector:
        if isinstance(other, int):
            return Vector([v << other for v in self._vector], self._n_dimentions)
        return NotImplemented

    def __neg__(self) -> Vector:
        return Vector([-v for v in self._vector], self._n_dimentions)

    def __pos__(self) -> Vector:
        return Vector(self._vector.copy(), self._n_dimentions)

    def __invert__(self) -> Vector:
        return Vector([~v for v in self._vector], self._n_dimentions)

    def __iadd__(self, other: Vector | int | float) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions:
                for i, _ in enumerate(self._vector):
                    self._vector[i] += other._vector[i]
                return self
            raise ValueError
        if isinstance(other, (int, float)):
            for i, _ in enumerate(self._vector):
                self._vector[i] += other
            return self
        raise ValueError

    def __isub__(self, other: Vector | int | float) -> Vector:
        if isinstance(other, Vector):
            if self._n_dimentions == other._n_dimentions:
                for i, _ in enumerate(self._vector):
                    self._vector[i] -= other._vector[i]
                return self
            raise ValueError
        if isinstance(other, (int, float)):
            for i, _ in enumerate(self._vector):
                self._vector[i] -= other
            return self
        raise ValueError

    def __imul__(self, other: int | float) -> Vector:
        if isinstance(other, (int | float)):
            for i, _ in enumerate(self._vector):
                self._vector[i] *= other
        raise ValueError

    def __itruediv__(self, other: int | float) -> Vector:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._vector):
                    self._vector[i] /= other
        raise ValueError

    def __imod__(self, other: int | float) -> Vector:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._vector):
                    self._vector[i] %= other
        raise ValueError

    def __ifloordiv__(self, other: int | float) -> Vector:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._vector):
                    self._vector[i] //= other
        raise ValueError

    def __ipow__(self, other: int | float) -> Vector:
        if isinstance(other, (int | float)):
            for i, _ in enumerate(self._vector):
                self._vector[i] **= other
        raise ValueError

    def __iand__(self, other: int) -> Vector:
        if isinstance(other, int):
            for i, _ in enumerate(self._vector):
                self._vector[i] &= other
        raise ValueError

    def __ior__(self, other: int) -> Vector:
        if isinstance(other, int):
            for i, _ in enumerate(self._vector):
                self._vector[i] |= other
        raise ValueError

    def __ixor__(self, other: int) -> Vector:
        if isinstance(other, int):
            for i, _ in enumerate(self._vector):
                self._vector[i] ^= other
        raise ValueError

    def __irshift__(self, other: int) -> Vector:
        if isinstance(other, int):
            for i, _ in enumerate(self._vector):
                self._vector[i] >>= other
        raise ValueError

    def __ilshift__(self, other: int) -> Vector:
        if isinstance(other, int):
            for i, _ in enumerate(self._vector):
                self._vector[i] <<= other
        raise ValueError

    def __getattribute__(self, name: str) -> Any:
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        mapping = {"x": 0, "y": 1, "z": 2}
        if name in mapping:
            return self._vector[mapping[name]]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_n_dimentions":
            if hasattr(self, "_n_dimentions"):
                raise AttributeError("read only attribute")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_vector":
            raise AttributeError("cannot delete core data")
        object.__delattr__(self, name)

    def __dir__(self) -> list[str]:
        return super().__dir__() + ["x", "y", "z"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    class _AsyncVectorIter:
        def __init__(self, vector: Sequence[float], n_dimentions: int) -> None:
            self._n_dimentions = n_dimentions
            self._vector = vector
            self._i = 0

        def __aiter__(self) -> Vector._AsyncVectorIter:
            return self

        async def __anext__(self) -> float:
            if self._i >= self._n_dimentions:
                raise StopAsyncIteration

            await asyncio.sleep(0)
            value = self._vector[self._i]
            self._i += 1
            return value

    def __aiter__(self) -> _AsyncVectorIter:
        return self._AsyncVectorIter(self._vector, self._n_dimentions)

    def __await__(self):
        async def _wrap():
            return self
        return _wrap().__await__()
