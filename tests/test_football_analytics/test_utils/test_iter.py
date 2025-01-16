import queue

from football_analytics.utils.iter import peek_and_restore


def test_peek_and_restore_with_empty_queue():
    test_queue = queue.Queue()
    result = peek_and_restore(test_queue, 5)
    assert result == []
    assert test_queue.qsize() == 0


def test_peek_and_restore_with_less_items_than_n():
    test_queue = queue.Queue()
    for i in range(3):
        test_queue.put(i)

    result = peek_and_restore(test_queue, 5)
    assert result == [0, 1, 2]
    assert test_queue.qsize() == 3
    assert list(test_queue.queue) == [0, 1, 2]


def test_peek_and_restore_with_exact_items_as_n():
    test_queue = queue.Queue()
    for i in range(5):
        test_queue.put(i)

    result = peek_and_restore(test_queue, 5)
    assert result == [0, 1, 2, 3, 4]
    assert test_queue.qsize() == 5
    assert list(test_queue.queue) == [0, 1, 2, 3, 4]


def test_peek_and_restore_with_more_items_than_n():
    test_queue = queue.Queue()
    for i in range(10):
        test_queue.put(i)

    result = peek_and_restore(test_queue, 5)
    assert result == [0, 1, 2, 3, 4]
    assert test_queue.qsize() == 10
    assert list(test_queue.queue) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_peek_and_restore_with_n_equal_zero():
    test_queue = queue.Queue()
    for i in range(3):
        test_queue.put(i)

    result = peek_and_restore(test_queue, 0)
    assert result == []
    assert test_queue.qsize() == 3
    assert list(test_queue.queue) == [0, 1, 2]


def test_peek_and_restore_with_n_negative():
    test_queue = queue.Queue()
    for i in range(3):
        test_queue.put(i)

    result = peek_and_restore(test_queue, -1)
    assert result == []
    assert test_queue.qsize() == 3
    assert list(test_queue.queue) == [0, 1, 2]
