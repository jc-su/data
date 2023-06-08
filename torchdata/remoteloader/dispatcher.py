import numpy as np
import zmq
import redis


class Dispatcher:
    def __init__(self, filelist: np.array, cache: np.array, workers: dict):
        self.filelist = filelist
        self.cache = cache
        self.workers = workers
        self.memory_wss = 0.6 * len(self.filelist) // 1
        self.nvme_wss = 0.4 * self.memory_wss // 1

    def dispatch(self):
        # Determine cached and uncached files
        cached_files = [file for file in self.filelist if file in self.cache]
        uncached_files = [file for file in self.filelist if file not in self.cache]

        # Calculate total batch size
        total_batch_size = sum(worker['batch_size'] for worker in self.workers)

        # generate

        # Distribute uncached files to workers based on their batch sizes
        for worker in self.workers:
            weight = worker['batch_size'] / total_batch_size
            num_files = int(weight * len(uncached_files))
            files_for_worker = uncached_files[:num_files]
            uncached_files = uncached_files[num_files:]

            # Send each file to worker and update cache when processed
            batch_files = np.array_split(files_for_worker, worker['batch_size'])
            print(batch_files)

    def hit(self):
        return self.dispatch()

    def miss(self):

        pass

    def dispatch_to_worker(self, worker, files):
        pass

    def dispatch_to_trainer(self, trainer, files):
        pass

    def eviction(self):
        pass


# Example usage:
if __name__ == '__main__':
    filelist = list(range(1, 51))
    cached_files = [1, 4, 6, 7, 10, 16, 19, 25, 31, 33, 44, 48]
    workers = [
        {'id': 1, 'batch_size': 3},
        {'id': 2, 'batch_size': 2},
        {'id': 3, 'batch_size': 5},
    ]

    dispatcher = Dispatcher(filelist, set(cached_files), workers)
    print(dispatcher.nvme_wss, dispatcher.memory_wss)
