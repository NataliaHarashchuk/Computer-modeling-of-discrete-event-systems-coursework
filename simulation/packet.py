from dataclasses import dataclass, field


@dataclass
class NodeRecord:
    node_name: str
    arrival_time: float
    queue_exit_time: float | None = None
    service_start_time: float | None = None
    accelerated: bool | None = None

    @property
    def waiting_time(self):
        if self.queue_exit_time is None:
            return None
        return self.queue_exit_time - self.arrival_time


@dataclass
class Packet:
    id: int
    arrival_time: float
    exit_time: float | None = None
    destroyed: bool | None = None
    discarded: bool = False
    node_records: list[NodeRecord] = field(default_factory=list)

    @property
    def sojourn_time(self):
        if self.exit_time is None:
            return None
        return self.exit_time - self.arrival_time

    @property
    def was_accelerated(self):
        return any(record.accelerated for record in self.node_records)

    def open_node_record(self, node_name, arrival_time):
        record = NodeRecord(node_name=node_name, arrival_time=arrival_time)
        self.node_records.append(record)
        return record

    def node_record(self, node_name):
        for record in self.node_records:
            if record.node_name == node_name:
                return record
        return None
