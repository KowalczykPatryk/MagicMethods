from setuptools import setup, Extension

if __name__ == "__main__":
    setup(name="vector_buffer",
        version="1.0.0",
        description="Implementation of Python's buffer protocol",
        ext_modules=[Extension("vector_buffer", ["vector_buffer.c"])])
