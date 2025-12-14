from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MoveDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOVE_FORWARD: _ClassVar[MoveDirection]
    MOVE_BACKWARD: _ClassVar[MoveDirection]
    MOVE_LEFT: _ClassVar[MoveDirection]
    MOVE_RIGHT: _ClassVar[MoveDirection]
MOVE_FORWARD: MoveDirection
MOVE_BACKWARD: MoveDirection
MOVE_LEFT: MoveDirection
MOVE_RIGHT: MoveDirection

class MoveRequest(_message.Message):
    __slots__ = ("direction", "duration")
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    direction: MoveDirection
    duration: float
    def __init__(self, direction: _Optional[_Union[MoveDirection, str]] = ..., duration: _Optional[float] = ...) -> None: ...

class KeyInput(_message.Message):
    __slots__ = ("key_value",)
    KEY_VALUE_FIELD_NUMBER: _ClassVar[int]
    key_value: str
    def __init__(self, key_value: _Optional[str] = ...) -> None: ...
