import multiprocessing as mp

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class Arx5InterpolationController(mp.Process):
    pass
    @property


    def is_ready(self):
        return self.ready_event.is_set()
    

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5InterpolationController] Controller process spawned at {self.pid}")

    
    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self):
        ...

    def get_gripper_state(self):
        ...
    
    def get_all_state(self):
        ...

    def get_all_gripper_state(self):
        ...

    def schedule_waypoint(self):
        ...

    def schedule_gripper_waypoint(self):
        ...

    