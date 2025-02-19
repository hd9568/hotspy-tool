class Function:
    def __init__(self, type_, max_buf, visits, time, time_percent, time_per_visit, region):
        self.type = type_
        self.max_buf = int(max_buf.replace(",", ""))
        self.visits = int(visits.replace(",", ""))
        self.time = float(time)
        self.time_percent = float(time_percent)
        self.time_per_visit = float(time_per_visit)
        self.region = region
        if(self.time_percent>=0.1 and type_=="USR"):
            self.isHotpot = 1
        else:
            self.isHotpot = 0
    
    def __repr__(self):
        return (f"Function(type={self.type}, max_buf={self.max_buf}, visits={self.visits}, "
                f"time={self.time}, time_percent={self.time_percent}, "
                f"time_per_visit={self.time_per_visit}, region='{self.region}')")
    
    def get_type(self):
        return self.type

    def get_max_buf(self):
        return self.max_buf

    def get_visits(self):
        return self.visits

    def get_time(self):
        return self.time

    def get_time_percent(self):
        return self.time_percent

    def get_time_per_visit(self):
        return self.time_per_visit

    def get_region(self):
        return self.region
    
    def get_hotpot(self):
        return self.isHotpot
