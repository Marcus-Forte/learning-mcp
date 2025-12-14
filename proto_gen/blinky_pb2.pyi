from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SetLEDRequest(_message.Message):
    __slots__ = ("line", "on")
    LINE_FIELD_NUMBER: _ClassVar[int]
    ON_FIELD_NUMBER: _ClassVar[int]
    line: int
    on: bool
    def __init__(self, line: _Optional[int] = ..., on: bool = ...) -> None: ...

class IsOnRequest(_message.Message):
    __slots__ = ("line",)
    LINE_FIELD_NUMBER: _ClassVar[int]
    line: int
    def __init__(self, line: _Optional[int] = ...) -> None: ...

class IsOnResponse(_message.Message):
    __slots__ = ("is_on",)
    IS_ON_FIELD_NUMBER: _ClassVar[int]
    is_on: bool
    def __init__(self, is_on: bool = ...) -> None: ...
