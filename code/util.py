def load_dataset(f1, f2, f3, batch_size):
    fd1, fd2, fd3 = open(f1), open(f2), open(f3)
    batch = []
    for _ in range(size_of_dataset:
        line1, line2, line3 = fd1.readline(), fd2.readline(), fd3, readline()
        batch.append((line1, line2, line3))

        if len(batch,) == batch_size:
            yield batch
