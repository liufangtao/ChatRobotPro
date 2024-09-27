import concurrent.futures  
import queue  
import threading  
  
class SingleWorkerExecutor:  
    def __init__(self):  
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  
        self._result_queue = queue.Queue()  
        self._lock = threading.Lock()  
  
    def submit(self, func, *args, **kwargs):  
        future = self._executor.submit(func, *args, **kwargs)  
          
        # 使用锁来同步对结果队列的访问  
        with self._lock:  
            # 将future的引用放入结果队列，而不是结果本身  
            # 因为我们想要的是能够在future完成时获取结果  
            self._result_queue.put(future)  
          
        return future  
  
    def result(self):  
        with self._lock:  
            # 从结果队列中获取future对象  
            future = self._result_queue.get()  
            # 等待future完成并获取结果  
            result = future.result()  
        return result  
  
    def shutdown(self):  
        self._executor.shutdown(wait=True)  