import time
from concurrent.futures import ThreadPoolExecutor
from modules.util.logger import Logger
from collections.abc import Callable

import threading


g_logger = Logger()
g_lock = threading.Lock()


class MultiThreader(object):
    def __init__(self, n_workers=None, name_prefix="MultiThreader"):
        self.n_workers = n_workers
        self.name_prefix = name_prefix
        self.params = dict(max_workers=self.n_workers, thread_name_prefix=name_prefix)
        g_logger.info(name_prefix)

    def do_execute(self, worker, tasks):
        g_logger.info("do_execute() Start", str(worker), str(tasks))
        with ThreadPoolExecutor(**self.params) as executor:
            g_logger.info("with statement Start", str(worker), str(tasks))
            futures = [executor.submit(worker, tsk) for tsk in tasks]
            g_logger.info("with statement Result:" + "|".join([str(f.result()) for f in futures]))
            g_logger.info("with statement End", str(worker), str(tasks))
        g_logger.info("do_execute() End", str(worker), str(tasks))
        return futures


if __name__ == "__main__":

    def worker(task):
        g_logger.info("worker() Start", task)
        # with g_lock:
        #     g_logger.info("worker.g_lock Start", task)
        #     time.sleep(1.111)
        #     g_logger.info("worker.g_lock End", task)

        def subworker(subtask):
            g_logger.info("subworker() Start", subtask)
            # with g_lock:
            #     g_logger.info("subworker.g_lock Start", subtask)
            #     time.sleep(0.3)
            #     g_logger.info("subworker.g_lock End", subtask)

            def subsubworker(subsubtask):
                g_logger.info("subsubworker() Start", subsubtask)
                # with g_lock:
                g_logger.info("subsubworker.g_lock Start", subsubtask)
                time.sleep(0.3)
                g_logger.info("subsubworker.g_lock End", subsubtask)

            subsubtasks = [ f"{subtask}_sub{idx:02d}" for idx in range(5) ]
            subsubmlt = MultiThreader(n_workers=None, name_prefix="MultiGrandChild")
            submlt.do_execute(subsubworker, subsubtasks)

            g_logger.info("subworker() End", subtask)
            return

        subtasks = [ f"{task}_sub{idx:02d}" for idx in range(5) ]
        submlt = MultiThreader(n_workers=None, name_prefix="MultiChild")
        submlt.do_execute(subworker, subtasks)

        g_logger.info("worker() End", task)
        return


    mlt = MultiThreader(n_workers=2, name_prefix="MultiParent")
    tasks = [ f"task{idx:02d}" for idx in range(3) ]
    mlt.do_execute(worker, tasks)
