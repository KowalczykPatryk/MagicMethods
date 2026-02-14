from linalg import vector_buffer


if __name__ == "__main__":
    v = vector_buffer.Vector(5)
    m = memoryview(v)
