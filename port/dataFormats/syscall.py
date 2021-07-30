from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from base.DataFormatter import DataFormatter, EventTypeClassifier
from misc.Utils import str_to_number

SYSCALL_KEY = "SysCall"
PROBABILITY_KEY = "Probability"

SEA_COLUMN_KEYS=[
	"SysCall",
	"File Handle"
]

ADDITIONAL_OPTIONAL_KEYS = [PROBABILITY_KEY]

current_time=datetime.now()
time_increment=timedelta(microseconds=1)
line_number=1   # record the line number in the input stream


class SysCallEventTypeClassifier(EventTypeClassifier):
    """
    This type classifier assigns a dedicated event type to each syscall.
    """
    def get_event_type(self, event_payload: dict):
        """
        The type of a stock event is equal to the stock ticker (company name).
        """
        return event_payload[SYSCALL_KEY]


class SysCallDataFormatter(DataFormatter):
    """
    A data formatter implementation of basic system calls.
    """
    def __init__(self, event_type_classifier: EventTypeClassifier = SysCallEventTypeClassifier()):
        super().__init__(event_type_classifier)

    def parse_event(self, raw_data: str):
        event_attributes = raw_data.replace("\n", "").split(",")
        parsed_events=dict(zip(
            SEA_COLUMN_KEYS,
            map(str_to_number, event_attributes)
        ))
        global line_number
        parsed_events["Line#"]=line_number
        line_number+=1
        return parsed_events
        """
        return dict(zip(
            SEA_COLUMN_KEYS,
            map(str_to_number, event_attributes)
        ))
        """

    def get_event_timestamp(self, event_payload: dict):
        """
        The event timestamp is essentially just a microsecond counter starting at time of first run
        """
        global current_time
        current_time=current_time+time_increment
        return current_time

    def get_probability(self, event_payload: Dict[str, Any]) -> Optional[float]:
        return event_payload.get(PROBABILITY_KEY, None)
