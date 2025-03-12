import asyncio
import random
from typing import List


async def test(id: str, n: int) -> List[int]:
    sec = random.randint(1, 10)
    print(f"[{id}] start. sec = {sec}")
    for i in range(1, n + 1):
        print(f"[{id}] => {i}")
        await asyncio.sleep(sec)
    print(f"[{id}] end.")


async def main():
    await asyncio.gather(
        test("A", 5),
        test("B", 7),
        test("C", 11),
    )


if __name__ == "__main__":
    asyncio.run(main())
