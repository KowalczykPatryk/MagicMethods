"""
This module is container for Matrix class
"""

from __future__ import annotations
import struct
from collections.abc import Iterator
from typing import Sequence, Any
import asyncio
import copy
from .vector import Vector


class Matrix:
    """
    Represents a matrix.
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

    def __init__(self, init_list: list[list[float]], n_rows: int, n_cols: int) -> None:
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError
        self._matrix: list = init_list
        for i in range(len(init_list), n_rows):
            for j in range(len(init_list[i]), n_cols):
                self._matrix[i][j] = 0
        self._n_rows: int = n_rows
        self._n_cols: int = n_cols
        self._shape: tuple = (n_rows, n_cols)

    def __repr__(self) -> str:
        matrix_str = ""
        for i in range(self._matrix):
            for j in range(self._matrix[i]):
                matrix_str += str(self._matrix[i][j])
                matrix_str += ", "
            matrix_str += "\n"
        return f"Matrix(\n{matrix_str}\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._matrix == other._matrix

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._matrix)

    def matrix_norm(self) -> float:
        """
        Calculates the 1-norm (column norm) of the matrix induced from 
        vector norm (https://en.wikipedia.org/wiki/Matrix_norm)

        :param self: 
        :return: norm of the matrix
        :rtype: float
        """
        column_sum = [0*len(self._matrix[0])]
        for i in range(self._matrix[0]):
            for j in range(self._matrix):
                column_sum[i] += self._matrix[j][i]
        return max(column_sum)

    def __lt__(self, other: Matrix) -> bool:
        return self.matrix_norm() < other.matrix_norm()

    def __gt__(self, other: Matrix) -> bool:
        return not self.__lt__(other)

    def __le__(self, other: Matrix) -> bool:
        return self.matrix_norm() <= other.matrix_norm()

    def __ge__(self, other: Matrix) -> bool:
        return not self.__le__(other)

    def __str__(self) -> str:
        return (
            self.__repr__() +
            f"Has {self._n_rows} {'row' if self._n_rows == 1 else 'rows'} and "
            f"{self._n_cols} {'column' if self._n_cols == 1 else 'columns'}"
        )

    def __bool__(self) -> bool:
        return any(any(row) for row in self._matrix)

    def __int__(self) -> int:
        if self._n_rows == 1 and self._n_cols == 1:
            return int(self._matrix)
        raise TypeError("Cannot convert multidimentional Matrix to int")

    def __float__(self) -> float:
        if self._n_rows == 1 and self._n_cols == 1:
            return float(self._matrix)
        raise TypeError("Cannot convert multidimentional Matrix to float")

    def __bytes__(self) -> bytes:
        return (
            struct.pack("I", self._n_rows) +
            struct.pack("I", self._n_cols) +
            "".join(struct.pack("d", row) for row in self._matrix).encode()
        )

    def __complex__(self) -> complex:
        return NotImplemented

    def __format__(self, spec: str) -> str:
        if not spec:
            return str(self._matrix)
        return (
            "[\n" +
            "\n".join(", ".join(format(v, spec) for v in row) for row in self._matrix) +
            "\n]"
        )

    class _PrecisionContext:
        def __init__(self, matrix_object: Vector, precision: int) -> None:
            self._matrix_object: Vector = matrix_object
            self._precision: int = precision
            self._old_precision: int = None

        def __enter__(self) -> Vector:
            self._old_precision = self._precision
            self._matrix_object._precision = self._precision
            return self._matrix_object

        def __exit__(self, exc_type, exc, traceback) -> bool:
            self._matrix_object._precision = self._old_precision

    def precision(self, precision: int) -> _PrecisionContext:
        """
        Docstring for precision

        :param self: This instance of the class Matrix.
        :param precision: Precision for each number stored inside Matrix.
        :type precision: int
        :return: Object that handles context managing used when "with" syntax.
        :rtype: _PrecisionContext
        """
        return self._PrecisionContext(self, precision)

    def __len__(self) -> int:
        return self._n_rows

    class _MatrixIterator(Iterator[float]):
        """
            Matrix iterator iterates over matrix's rows
        """

        def __init__(self, matrix: Sequence[Sequence[float]]) -> None:
            self.current_row_index = 0
            self.matrix = matrix

        def __iter__(self) -> Matrix._MatrixIterator:
            return self

        def __next__(self) -> float:
            current = self.current_row_index
            self.current_row_index += 1
            if current >= len(self.matrix):
                raise StopIteration
            return self.matrix[current]

        def __length_hint__(self):
            return len(self.matrix) - self.current_row_index

    def __iter__(self) -> _MatrixIterator:
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
        :rtype: MatrixIterator
        """
        return self._MatrixIterator(self._matrix)

    def __getitem__(self, index: int | tuple) -> float | Vector | Matrix:
        if isinstance(index, int):
            return Vector(self._matrix[index], len(self._matrix[index]))
        if isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, int):
                return self._matrix[row][col]

            if isinstance(row, slice) and isinstance(col, int):
                return Vector([rows[col] for rows in self._matrix[row]], len(self._matrix[row]))

            if isinstance(row, int) and isinstance(col, slice):
                return Vector(self._matrix[row][col], len(self._matrix[row][col]))

            if isinstance(row, slice) and isinstance(col, slice):
                return Matrix(
                    [rows[col] for rows in self._matrix[row]],
                    len(self._matrix[row]), len(self._matrix[0][col])
                )
        raise TypeError("Invalid Index")

    def __setitem__(self, index: int | tuple,
                    value: float | list[float] | list[list[float]]) -> None:
        if isinstance(index, int):
            if isinstance(value, float):
                self._matrix[index] = value
                return
            raise ValueError("Can't change size of matrix")

        if isinstance(index, tuple):
            row, col = index

            if isinstance(row, int) and isinstance(col, int):
                self._data[row][col] = value
                return

            if isinstance(row, slice) and isinstance(col, int):
                if len(self._matrix[row]) == len(value):
                    start, stop, step = row
                    indices = len(range(start, stop, step))
                    for i, j in zip(range(start, stop, step), indices):
                        self._matrix[i][col] = value[j]
                raise ValueError("Can't change size of matrix")

            if isinstance(row, int) and isinstance(col, slice):
                if len(self._matrix[row][col]) == len(value):
                    self._matrix[row][col] = value
                raise ValueError("can't change size of matrix")

            if isinstance(row, slice) and isinstance(col, slice):
                if (len(self._matrix[row]) == len(value) and
                    len(self._matrix[row][col]) == len(value[0])):
                    start, stop, step = row
                    indices = len(range(start, stop, step))
                    for i, j in zip(range(start, stop, step), indices):
                        self._matrix[i][col] = value[j][col]
                raise ValueError("Can't change size of matrix")

        raise NotImplementedError("Slice assignment not implemented")

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("Matrix size cannot be changed")

    def __contains__(self, value: float) -> bool:
        return any(value in row for row in self._matrix)

    def __reversed__(self) -> Iterator[float]:
        return reversed(self._matrix)

    def __call__(self, other: Vector) -> Vector:
        if isinstance(other, Vector):
            if other._n_dimentions != self._n_cols:
                raise ValueError("Dimension mismatch")

            result = []
            for row in self._matrix:
                result.append(sum(a * b for a, b in zip(row, other)))

            return Vector(result, len(result))
        raise ValueError("You can only call matrix with vector")

    def __add__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, Matrix):
            if self._n_rows != other._n_rows and self._n_cols != other._n_cols:
                raise ValueError("Matrix shape mismatch")
            return Matrix(
                [[a+b for a, b in zip(row_self, row_other)]
                for row_self, row_other in zip(self._matrix, other._matrix)],
                self._n_rows,
                self._n_cols
            )
        if isinstance(other, (int, float)):
            return Matrix(
                [[v+ other for v in row] for row in self._matrix], self._n_rows, self._n_cols
            )
        raise ValueError("To matrix you can only add other matrix or int, float")

    def __sub__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, Matrix):
            if self._n_rows != other._n_rows and self._n_cols != other._n_cols:
                raise ValueError("Matrix shape mismatch")
            return Matrix(
                [[a-b for a, b in zip(row_self, row_other)]
                for row_self, row_other in zip(self._matrix, other._matrix)],
                self._n_rows,
                self._n_cols
            )
        if isinstance(other, (int, float)):
            return Matrix(
                [[v+ other for v in row] for row in self._matrix], self._n_rows, self._n_cols
            )
        raise ValueError("To matrix you can only subtract other matrix or int, float")

    def __mul__(self, other: Vector | int | float | Matrix) -> Vector | Matrix:
        if isinstance(other, Vector):
            if other._n_dimentions != self._n_cols:
                raise ValueError("Dimension mismatch")

            result = []
            for row in self._matrix:
                result.append(sum(a * b for a, b in zip(row, other)))

            return Vector(result, len(result))
        if isinstance(other, (int, float)):
            return Matrix(
                [[v*other for v in row] for row in self._matrix], self._n_rows, self._n_cols
            )
        if isinstance(other, Matrix):
            if self._n_cols != other._n_rows:
                raise ValueError("Dimension mismatch")

            other_cols = list(zip(*other._matrix))

            return Matrix([
                [sum(a*b for a,b in zip(row, col)) for col in other_cols]
                for row in self._matrix
            ],
            self._n_rows,
            other._n_cols
            )
        return NotImplemented

    def __rmul__(self, other: Vector | int | float | Matrix) -> Vector | Matrix:
        if isinstance(other, (int, float, Vector, Matrix)):
            return self * other
        return NotImplemented

    def __truediv__(self, divisor: int | float) -> Matrix:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Matrix(
            [[v/divisor for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __mod__(self, divisor: int | float) -> Matrix:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Matrix(
            [[v%divisor for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __floordiv__(self, divisor: int | float) -> Matrix:
        if not isinstance(divisor, (int, float)):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError
        return Matrix(
            [[v//divisor for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __pow__(self, power: int) -> Matrix:
        return Matrix(
            [[v**power for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __and__(self, other: Matrix | int) -> Matrix:
        if isinstance(other, Matrix):
            if self._shape == other._shape:
                return Matrix(
                    [[a&b for a,b in zip(row_self, row_other)]
                     for row_self, row_other in zip(self._matrix, other._matrix)],
                    self._n_rows,
                    self._n_cols
                )
            raise ValueError
        if isinstance(other, int):
            return Matrix(
                [[v&other for v in row] for row in self._matrix],
                self._n_rows,
                self._n_cols
            )
        return NotImplemented

    def __rand__(self, other: int) -> Matrix:
        return self.__and__(other)

    def __or__(self, other: Matrix | int) -> Matrix:
        if isinstance(other, Matrix):
            if self._shape == other._shape:
                return Matrix(
                    [[a|b for a,b in zip(row_self, row_other)]
                     for row_self, row_other in zip(self._matrix, other._matrix)],
                    self._n_rows,
                    self._n_cols
                )
            raise ValueError
        if isinstance(other, int):
            return Matrix(
                [[v|other for v in row] for row in self._matrix],
                self._n_rows,
                self._n_cols
            )
        return NotImplemented

    def __ror__(self, other: int) -> Matrix:
        return self.__or__(other)

    def __xor__(self, other: Matrix | int) -> Matrix:
        if isinstance(other, Matrix):
            if self._shape == other._shape:
                return Matrix(
                    [[a^b for a,b in zip(row_self, row_other)]
                     for row_self, row_other in zip(self._matrix, other._matrix)],
                    self._n_rows,
                    self._n_cols
                )
            raise ValueError
        if isinstance(other, int):
            return Matrix(
                [[v^other for v in row] for row in self._matrix],
                self._n_rows,
                self._n_cols
            )
        return NotImplemented

    def __rxor__(self, other: int) -> Matrix:
        return self.__xor__(other)

    def __rshift__(self, other: int) -> Matrix:
        if isinstance(other, int):
            return Matrix(
                [[v >> other for v in row] for row in self._matrix],
                self._n_rows,
                self._n_cols
            )
        return NotImplemented

    def __lshift__(self, other: int) -> Matrix:
        if isinstance(other, int):
            return Matrix(
                [[v << other for v in row] for row in self._matrix],
                self._n_rows,
                self._n_cols
            )
        return NotImplemented

    def __neg__(self) -> Matrix:
        return Matrix(
            [[-v for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __pos__(self) -> Matrix:
        return Matrix(copy.deepcopy(self._matrix), self._n_rows, self._n_cols)

    def __invert__(self) -> Matrix:
        return Matrix(
            [[~v for v in row] for row in self._matrix],
            self._n_rows,
            self._n_cols
        )

    def __iadd__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, Matrix):
            if self._shape == other._shape:
                for i, _ in enumerate(self._matrix):
                    for j, _ in enumerate(self._matrix[i]):
                        self._matrix[i][j] += other._matrix[i][j]
                return self
            raise ValueError("Matrix shape mismatch.")
        if isinstance(other, (int, float)):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] += other
            return self
        raise ValueError

    def __isub__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, Matrix):
            if self._shape == other._shape:
                for i, _ in enumerate(self._matrix):
                    for j, _ in enumerate(self._matrix[i]):
                        self._matrix[i][j] -= other._matrix[i][j]
                return self
            raise ValueError
        if isinstance(other, (int, float)):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] -= other
            return self
        raise ValueError

    def __imul__(self, other: int | float) -> Matrix:
        if isinstance(other, (int | float)):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] *= other
        raise ValueError

    def __itruediv__(self, other: int | float) -> Matrix:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._matrix):
                    for j, _ in enumerate(self._matrix[i]):
                        self._matrix[i][j] /= other
        raise ValueError

    def __imod__(self, other: int | float) -> Matrix:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._matrix):
                    for j, _ in enumerate(self._matrix[i]):
                        self._matrix[i][j] %= other
        raise ValueError

    def __ifloordiv__(self, other: int | float) -> Matrix:
        if isinstance(other, (int | float)):
            if other != 0:
                for i, _ in enumerate(self._matrix):
                    for j, _ in enumerate(self._matrix[i]):
                        self._matrix[i][j] //= other
        raise ValueError

    def __ipow__(self, other: int | float) -> Matrix:
        if isinstance(other, (int | float)):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] **= other
        raise ValueError

    def __iand__(self, other: int) -> Matrix:
        if isinstance(other, int):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] &= other
        raise ValueError

    def __ior__(self, other: int) -> Matrix:
        if isinstance(other, int):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] |= other
        raise ValueError

    def __ixor__(self, other: int) -> Matrix:
        if isinstance(other, int):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] ^= other
        raise ValueError

    def __irshift__(self, other: int) -> Matrix:
        if isinstance(other, int):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] >>= other
        raise ValueError

    def __ilshift__(self, other: int) -> Matrix:
        if isinstance(other, int):
            for i, _ in enumerate(self._matrix):
                for j, _ in enumerate(self._matrix[i]):
                    self._matrix[i][j] <<= other
        raise ValueError

    def __getattribute__(self, name: str) -> Any:
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Vector:
        """
        Columns aren't stored directly, so m.c0 accesses 
        first column of matrix and returns it as Vector
        """
        if name.startswith("c") and name[1:].isdigit():
            idx = int(name[1:])
            if 0 <= idx < self._n_cols:
                return Vector([row[idx] for row in self._matrix], self._n_rows)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("_shape", "_n_rows", "_n_cols"):
            raise AttributeError("read only attribute")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_matrix":
            raise AttributeError("cannot delete core data")
        object.__delattr__(self, name)

    def __dir__(self) -> list[str]:
        return super().__dir__() + [f"c{col}" for col in range(self._n_cols)]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    class _AsyncMatrixIter:
        def __init__(self, matrix: Sequence[Sequence[float]], n_rows: int, n_cols: int) -> None:
            self._n_rows = n_rows
            self._n_cols = n_cols
            self._matrix = matrix
            self._i = 0

        def __aiter__(self) -> Matrix._AsyncMatrixIter:
            return self

        async def __anext__(self) -> float:
            if self._i >= self._n_rows:
                raise StopAsyncIteration

            await asyncio.sleep(0)
            value = self._matrix[self._i]
            self._i += 1
            return value

    def __aiter__(self) -> _AsyncMatrixIter:
        return self._AsyncMatrixIter(self._matrix, self._n_rows, self._n_cols)

    def __await__(self):
        async def _wrap():
            return self
        return _wrap().__await__()
