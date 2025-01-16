"""Utility functions for iterating over data structures."""

import queue


def peek_and_restore(peeked_queue: queue.Queue, n: int) -> list:
    """Peek the first n items from the queue and restore them in the same order.

    Args:
        queue (Queue): The queue to peek into.
        n (int): Number of items to peek.

    Returns:
        list: A list containing the first n items from the queue.
    """
    temp_queue: queue.Queue = queue.Queue()
    items = []

    for _ in range(n):
        if peeked_queue.empty():
            break
        item = peeked_queue.get()
        items.append(item)
        temp_queue.put(item)

    while not temp_queue.empty():
        peeked_queue.put(temp_queue.get())

    for _ in range(peeked_queue.qsize() - len(items)):
        peeked_queue.put(peeked_queue.get())

    return items
