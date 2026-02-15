from linalg import vector_buffer, Vector
import asyncio

async def main():
    async with Vector():
        print("inside")

asyncio.run(main())


if __name__ == "__main__":
    v = vector_buffer.Vector(5)
    m = memoryview(v)
