import threading
class Thread(object):
    condition_object = threading.Condition()
    def __init__(self, threadName):
        self.m_isRunning = False
        self.m_isThreadWaiting = False
        self.m_name = threadName
        self.thr = None
        self.condition_object = threading.Condition()
    def threadProc(self, obj):
        obj.run()
    def start(self):
        self.m_isThreadWaiting = False
        self.m_isRunning = True
        self.thr = threading.Thread(target=self.threadProc, args=(self,))
        self.thr.start()
        return self.thr
    
    def isRunning(self):
        res = True
        self.condition_object.acquire()
        res = self.m_isRunning
        self.condition_object.release()
        return res

    def wait(self):
        self.m_isThreadWaiting = True
        self.isWait()

    def nofity(self):
        if self.m_isThreadWaiting is True:
            self.condition_object.acquire()
            self.m_isThreadWaiting = False
            self.condition_object.notify()
            self.condition_object.release()

    def stop(self):
        self.condition_object.acquire()
        self.m_isRunning = False
        self.condition_object.release()
        self.nofity()


    def isWait(self):
        if self.m_isThreadWaiting is True:
            self.condition_object.acquire()
            self.condition_object.wait()
            self.condition_object.release()
        else:
            self.condition_object.acquire()
            result = self.m_isRunning
            self.condition_object.release()
            return result
        return True