#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: mrichter
"""
from __future__ import annotations
import datetime
import enum
import json
from typing import List, Dict
from . import interfaces


class ReportCode(enum.Enum):
    INFO = 20
    INFO_PARSING = 21
    INFO_CONVERTING = 22
    INFO_REPAIRING = 23
    WARNING = 30
    WARNING_PARSING = 31
    WARNING_CONVERTING = 32
    WARNING_REPAIRING = 33
    ERROR = 40
    ERROR_PARSING = 41
    ERROR_CONVERTING = 42
    ERROR_REPAIRING = 43
    EXCEPTION = 50
    EXCEPTION_PARSING = 51
    EXCEPTION_CONVERTING = 52
    EXCEPTION_REPAIRING = 53

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid value for ReportCode")


class LogLevel(enum.Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    EXCEPTION = 'EXCEPTION'

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid value for LogLevel")


class Report:
    def __init__(self, message: str, code: ReportCode, level: LogLevel, timestamp: datetime.datetime = None):
        self.timestamp = datetime.datetime.now()
        if timestamp is not None:
            self.timestamp = timestamp
        self.message = message
        self.code = code
        self.level = level

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "code": self.code.value,
            "level": self.level.value
        }


class ReportContainer(interfaces.ReportContainer):
    def __init__(self, logs: List[Report] = None):
        if logs is not None and isinstance(logs, List):
            self.logs = logs
        else:
            self.logs = []
    
    def add_log(self, log: Report):
        self.logs.append(log)

    def get_logs(self) -> List[Report]:
        return self.logs

    def extend_log(self, report_container: ReportContainer):
        self.logs.extend(report_container.get_logs())

    def serialize(self, path_to_disk: str):
        with open(path_to_disk, 'w', encoding='UTF-8') as file:
            json.dump(obj=[log.to_dict() for log in self.logs], fp=file, indent=2)
