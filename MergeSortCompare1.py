import math
import multiprocessing
import random
import time
import matplotlib.pyplot as plt
import numpy as np


def merge(*args):
    left, right = args[0] if len(args) == 1 else args
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged


def merge_sort(data):
    length = len(data)
    if length <= 1:
        return data
    middle = length // 2
    left = merge_sort(data[:middle])
    right = merge_sort(data[middle:])
    return merge(left, right)


def merge_sort_parallel(data, cpu_count):

    #processes = multiprocessing.cpu_count()
    processes = cpu_count
    pool = multiprocessing.Pool(processes=processes)
    size = int(math.ceil(float(len(data)) / processes))
    data = [data[i * size:(i + 1) * size] for i in range(processes)]
    data = pool.map(merge_sort, data)
    while len(data) > 1:
        extra = data.pop() if len(data) % 2 == 1 else None
        data = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]
        data = pool.map(merge, data) + ([extra] if extra else [])
    return data[0]



if __name__ == "__main__":
    time_seq = 0
    data_unsorted = randomlist = random.sample(range(0, 3000000), 1000000)
    #count_core = multiprocessing.cpu_count()
    count_core = 16
    x_core = np.arange(start=1, stop=count_core + 1)
    y_time = np.array([])
    speed_up = np.array([])
    plt.xlabel("Core Count")
    plt.ylabel("Time(Seconds)")
    for i_c in range(1, count_core + 1):
        if i_c == 1:
            start_seq = time.time()
            data_sorted = merge_sort(data_unsorted)
            time_seq = time.time() - start_seq
        start = time.time()
        data_sorted = merge_sort_parallel(data_unsorted, i_c)
        # print(data_sorted)
        end = time.time() - start
        y_time = np.insert(y_time, len(y_time), end)
        speed_up = np.insert(speed_up, len(speed_up), time_seq / end)

    print(y_time)
    print(speed_up)
    plt.plot(x_core, y_time)
    print(time_seq)
    plt.plot(x_core, np.repeat(time_seq, count_core))
    plt.legend(['Parallel MergeSort', 'Sequential MergeSort'])
    plt.show()
    plt.xlabel("Core Count")
    plt.ylabel("Speed Up")
    plt.plot(x_core, speed_up)
    plt.show()


