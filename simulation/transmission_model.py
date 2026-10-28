import math
from dataclasses import asdict, dataclass, field

from simulation.config import ModelConfig
from simulation.controller import ResourceController
from simulation.create import Create
from simulation.destruction_limit_monitor import DestructionLimitMonitor
from simulation.early_discard import EarlyDiscard
from simulation.fun_rand import FunRand
from simulation.model import Model
from simulation.process import Process
from simulation.resource_state import ResourceState
from simulation.sink import Sink

PRIORITY_CHANNEL_2 = 0
PRIORITY_CHANNEL_1 = 1
PRIORITY_ARRIVAL = 2

MILLISECONDS_PER_SECOND = 1000.0
EXIT_EVENT_LABEL = "кінець обслуговування, вихід"


@dataclass
class ProtocolEntry:
    step: int
    time: float
    element_name: str
    event_label: str
    packet_id: int
    next_event_times: tuple
    queue_lengths: tuple
    channel_busy: tuple
    resource_accelerated: bool
    exited_packet_sojourn_time: float | None
    exited_packet_destroyed: bool | None
    window_destroyed: int | None
    window_length: int | None
    resource_switch: str


@dataclass
class TraceSnapshot:
    time: float
    observed_time: float
    exited: int
    destroyed: int
    destroyed_fraction: float
    activation_count: int
    accelerated_time: float
    mean_queue_length_1: float
    mean_queue_length_2: float
    utilization_1: float
    utilization_2: float


@dataclass
class SimulationResults:
    generated: int
    delivered: int
    destroyed: int
    discarded: int
    exited: int
    destroyed_fraction: float
    in_system: int
    end_time: float
    observation_time: float
    destroyed_per_second: float
    activation_count: int
    deactivation_count: int
    activations_per_second: float
    accelerated_time_fraction: float
    accelerated_packets_fraction: float
    limit_violation_time_fraction: float
    limit_violation_exits_fraction: float
    max_window_destroyed_fraction: float
    mean_queue_length_1: float
    mean_queue_length_2: float
    max_queue_length_1: int
    max_queue_length_2: int
    mean_waiting_time_1: float
    mean_waiting_time_2: float
    utilization_1: float
    utilization_2: float
    mean_sojourn_time: float
    throughput_per_second: float
    trace: list = field(default_factory=list)

    def to_dict(self):
        values = asdict(self)
        del values["trace"]
        return values


class TransmissionModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.rand = FunRand(config.seed)
        self.resource_state = ResourceState(accelerated=config.control_mode == "always")
        self.sink = Sink("Вихід", config.destruction_threshold, config.time_tolerance)
        self.create = Create(
            "Надходження", PRIORITY_ARRIVAL, self.rand, config.arrival_distribution,
            config.arrival_delay_mean, config.arrival_spread,
        )
        self.channel_1 = self.build_channel("K1", PRIORITY_CHANNEL_1)
        self.channel_2 = self.build_channel("K2", PRIORITY_CHANNEL_2)
        if config.early_discard:
            self.channel_1.early_discard = self.build_early_discard(self.channel_2)
            self.channel_2.early_discard = self.build_early_discard(None)
        self.create.next_element = self.channel_1
        self.channel_1.next_element = self.channel_2
        self.channel_2.next_element = self.sink
        self.processes = [self.channel_1, self.channel_2]
        self.event_sources = [self.create, self.channel_1, self.channel_2]
        self.model = Model([self.create, self.channel_1, self.channel_2, self.sink], config.time_tolerance)
        self.model.add_statistics_collector(self.resource_state)
        self.controller = self.build_controller()
        if self.controller is not None:
            self.sink.add_exit_listener(self.controller.on_packet_exit)
        self.limit_monitor = DestructionLimitMonitor(config.destruction_limit_window_size, config.destruction_limit)
        self.sink.add_exit_listener(self.limit_monitor.on_packet_exit)
        self.model.add_statistics_collector(self.limit_monitor)
        self.trace = []
        self.protocol = []
        self.accelerated_before_event = self.resource_state.accelerated
        if config.record_protocol:
            self.model.add_event_listener(self.record_protocol_entry)

    def build_channel(self, name, event_priority):
        config = self.config
        return Process(
            name, event_priority, self.rand, config.service_distribution, config.service_time_normal,
            config.service_time_accelerated, self.resource_state,
        )

    def build_early_discard(self, downstream_channel):
        config = self.config
        return EarlyDiscard(config.destruction_threshold, config.time_tolerance, self.sink, downstream_channel)

    def build_controller(self):
        if self.config.control_mode != "adaptive":
            return None
        return ResourceController(
            self.resource_state,
            self.config.window_size,
            self.config.activation_threshold,
            self.config.effective_deactivation_threshold,
        )

    def finish_warmup(self):
        self.model.reset_statistics()
        if self.controller is not None:
            self.controller.reset_statistics()

    def record_trace_snapshot(self):
        observed = self.model.observed_time
        exited = self.sink.exited_count
        self.trace.append(TraceSnapshot(
            time=self.model.current_time,
            observed_time=observed,
            exited=exited,
            destroyed=self.sink.destroyed_count,
            destroyed_fraction=self.sink.destroyed_count / exited if exited else 0.0,
            activation_count=self.controller.activation_count if self.controller else 0,
            accelerated_time=self.resource_state.accelerated_time,
            mean_queue_length_1=self.channel_1.mean_queue_length(observed),
            mean_queue_length_2=self.channel_2.mean_queue_length(observed),
            utilization_1=self.channel_1.utilization(observed),
            utilization_2=self.channel_2.utilization(observed),
        ))

    def record_protocol_entry(self, element, packet):
        accelerated_before = self.accelerated_before_event
        accelerated_after = self.resource_state.accelerated
        self.accelerated_before_event = accelerated_after
        if len(self.protocol) >= self.config.protocol_limit:
            return
        resource_switch = ""
        if accelerated_after and not accelerated_before:
            resource_switch = "ресурс УВІМКНЕНО"
        elif accelerated_before and not accelerated_after:
            resource_switch = "ресурс ВИМКНЕНО"
        packet_exited = element is self.channel_2
        self.protocol.append(ProtocolEntry(
            step=self.model.event_count,
            time=self.model.current_time,
            element_name=element.name,
            event_label=EXIT_EVENT_LABEL if packet_exited else element.event_label,
            packet_id=packet.id,
            next_event_times=tuple(event_source.next_time for event_source in self.event_sources),
            queue_lengths=tuple(process.queue_length for process in self.processes),
            channel_busy=tuple(process.is_busy for process in self.processes),
            resource_accelerated=accelerated_after,
            exited_packet_sojourn_time=packet.sojourn_time if packet_exited else None,
            exited_packet_destroyed=packet.destroyed if packet_exited else None,
            window_destroyed=self.controller.window.destroyed_count if self.controller else None,
            window_length=len(self.controller.window) if self.controller else None,
            resource_switch=resource_switch,
        ))

    def simulate(self):
        config = self.config
        if config.warmup_time > 0:
            self.model.run_until(config.warmup_time)
            self.finish_warmup()
        if config.trace_interval is not None:
            trace_index = 1
            trace_time = config.warmup_time + config.trace_interval
            while trace_time <= config.end_time:
                self.model.run_until(trace_time, include_events_at_stop_time=False)
                self.record_trace_snapshot()
                trace_index += 1
                trace_time = config.warmup_time + trace_index * config.trace_interval
        self.model.run_until(config.end_time)
        return self.collect_results()

    def collect_results(self):
        observation_time = self.config.simulation_time
        exited = self.sink.exited_count
        destroyed = self.sink.destroyed_count
        activation_count = self.controller.activation_count if self.controller else 0
        deactivation_count = self.controller.deactivation_count if self.controller else 0
        in_system = sum(process.queue_length + (1 if process.is_busy else 0) for process in self.processes)
        return SimulationResults(
            generated=self.create.created_count,
            delivered=self.sink.delivered_count,
            destroyed=destroyed,
            discarded=self.sink.discarded_count,
            exited=exited,
            destroyed_fraction=destroyed / exited if exited else 0.0,
            in_system=in_system,
            end_time=self.model.current_time,
            observation_time=observation_time,
            destroyed_per_second=destroyed / observation_time * MILLISECONDS_PER_SECOND,
            activation_count=activation_count,
            deactivation_count=deactivation_count,
            activations_per_second=activation_count / observation_time * MILLISECONDS_PER_SECOND,
            accelerated_time_fraction=self.resource_state.accelerated_time / observation_time,
            accelerated_packets_fraction=self.sink.accelerated_packets_count / exited if exited else 0.0,
            limit_violation_time_fraction=self.limit_monitor.violation_time / observation_time,
            limit_violation_exits_fraction=self.limit_monitor.violating_exits_fraction,
            max_window_destroyed_fraction=self.limit_monitor.max_destroyed_fraction,
            mean_queue_length_1=self.channel_1.mean_queue_length(observation_time),
            mean_queue_length_2=self.channel_2.mean_queue_length(observation_time),
            max_queue_length_1=self.channel_1.max_queue_length,
            max_queue_length_2=self.channel_2.max_queue_length,
            mean_waiting_time_1=self.channel_1.mean_waiting_time,
            mean_waiting_time_2=self.channel_2.mean_waiting_time,
            utilization_1=self.channel_1.utilization(observation_time),
            utilization_2=self.channel_2.utilization(observation_time),
            mean_sojourn_time=self.sink.mean_sojourn_time,
            throughput_per_second=exited / observation_time * MILLISECONDS_PER_SECOND,
            trace=list(self.trace),
        )

    def service_time_label(self, accelerated):
        service_time = self.config.service_time_accelerated if accelerated else self.config.service_time_normal
        return f"{service_time:g} мс"

    def format_protocol(self):
        process_names = [process.name for process in self.processes]
        next_time_headers = [f"t_next {source.name}" for source in self.event_sources]
        columns = (
            [("№", 5), ("Час, мс", 10), ("Елемент", -11), ("Подія", -28), ("Пакет", 5)]
            + [(header, 18 if index == 0 else 10) for index, header in enumerate(next_time_headers)]
            + [(f"Черга {name}", 8) for name in process_names]
            + [(f"Канал {name}", 8) for name in process_names]
            + [("Ресурс", 6), ("T пакета, мс", 12), ("Результат", -10), ("У вікні", 7), ("Примітка", 0)]
        )
        header = " ".join(format_protocol_cell(title, width) for title, width in columns)
        lines = [header, "-" * len(header)]
        for entry in self.protocol:
            cells = [entry.step, f"{entry.time:.4f}", entry.element_name, entry.event_label, entry.packet_id]
            cells += [format_event_time(next_time) for next_time in entry.next_event_times]
            cells += list(entry.queue_lengths)
            cells += ["зайнятий" if busy else "вільний" for busy in entry.channel_busy]
            cells.append(self.service_time_label(entry.resource_accelerated))
            cells.append("" if entry.exited_packet_sojourn_time is None else f"{entry.exited_packet_sojourn_time:.4f}")
            cells.append(format_exit_result(entry.exited_packet_destroyed))
            cells.append("" if entry.window_length is None else f"{entry.window_destroyed}/{entry.window_length}")
            cells.append(entry.resource_switch)
            lines.append(" ".join(format_protocol_cell(cell, width) for cell, (_, width) in zip(cells, columns)))
        return "\n".join(lines)


def format_protocol_cell(value, width):
    if width > 0:
        return f"{value:>{width}}"
    if width < 0:
        return f"{value:<{-width}}"
    return str(value)


def format_event_time(next_time):
    return "∞" if math.isinf(next_time) else f"{next_time:.4f}"


def format_exit_result(destroyed):
    if destroyed is None:
        return ""
    return "знищено" if destroyed else "доставлено"
