import math
from dataclasses import dataclass

EQUAL_RATES_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MM1Characteristics:
    arrival_rate: float
    service_rate: float
    utilization: float
    mean_queue_length: float
    mean_system_length: float
    mean_waiting_time: float
    mean_sojourn_time: float
    throughput: float


def mm1_characteristics(arrival_rate, service_rate):
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("rates must be positive")
    utilization = arrival_rate / service_rate
    if utilization >= 1.0:
        raise ValueError("queue is unstable: utilization must be below 1")
    mean_queue_length = utilization ** 2 / (1.0 - utilization)
    mean_sojourn_time = 1.0 / (service_rate - arrival_rate)
    mean_waiting_time = mean_sojourn_time - 1.0 / service_rate
    return MM1Characteristics(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        utilization=utilization,
        mean_queue_length=mean_queue_length,
        mean_system_length=utilization / (1.0 - utilization),
        mean_waiting_time=mean_waiting_time,
        mean_sojourn_time=mean_sojourn_time,
        throughput=arrival_rate,
    )


@dataclass(frozen=True)
class TandemCharacteristics:
    node_1: MM1Characteristics
    node_2: MM1Characteristics
    mean_sojourn_time: float
    mean_queue_length_total: float
    mean_system_length_total: float
    exponential_rate_1: float
    exponential_rate_2: float


def jackson_tandem_characteristics(arrival_rate, service_rate_1, service_rate_2):
    node_1 = mm1_characteristics(arrival_rate, service_rate_1)
    node_2 = mm1_characteristics(arrival_rate, service_rate_2)
    return TandemCharacteristics(
        node_1=node_1,
        node_2=node_2,
        mean_sojourn_time=node_1.mean_sojourn_time + node_2.mean_sojourn_time,
        mean_queue_length_total=node_1.mean_queue_length + node_2.mean_queue_length,
        mean_system_length_total=node_1.mean_system_length + node_2.mean_system_length,
        exponential_rate_1=service_rate_1 - arrival_rate,
        exponential_rate_2=service_rate_2 - arrival_rate,
    )


def hypoexponential_tail_probability(rate_1, rate_2, threshold):
    if rate_1 <= 0 or rate_2 <= 0:
        raise ValueError("rates must be positive")
    if threshold <= 0:
        return 1.0
    if abs(rate_1 - rate_2) <= EQUAL_RATES_TOLERANCE * max(rate_1, rate_2):
        return math.exp(-rate_1 * threshold) * (1.0 + rate_1 * threshold)
    return (rate_2 * math.exp(-rate_1 * threshold) - rate_1 * math.exp(-rate_2 * threshold)) / (rate_2 - rate_1)


def tandem_destruction_probability(arrival_rate, service_rate_1, service_rate_2, threshold):
    return hypoexponential_tail_probability(
        service_rate_1 - arrival_rate,
        service_rate_2 - arrival_rate,
        threshold,
    )
