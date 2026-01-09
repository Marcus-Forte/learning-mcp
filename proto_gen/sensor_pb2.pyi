from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CameraEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[CameraEncoding]
    RGB8: _ClassVar[CameraEncoding]
    BGR8: _ClassVar[CameraEncoding]
    GRAY8: _ClassVar[CameraEncoding]
    MJPEG: _ClassVar[CameraEncoding]
UNKNOWN: CameraEncoding
RGB8: CameraEncoding
BGR8: CameraEncoding
GRAY8: CameraEncoding
MJPEG: CameraEncoding

class Point3(_message.Message):
    __slots__ = ("x", "y", "z", "r", "g", "b", "intensity")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    INTENSITY_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    intensity: int
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., r: _Optional[float] = ..., g: _Optional[float] = ..., b: _Optional[float] = ..., intensity: _Optional[int] = ...) -> None: ...

class PointCloud3(_message.Message):
    __slots__ = ("points", "timestamp")
    POINTS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[Point3]
    timestamp: int
    def __init__(self, points: _Optional[_Iterable[_Union[Point3, _Mapping]]] = ..., timestamp: _Optional[int] = ...) -> None: ...

class IMUData(_message.Message):
    __slots__ = ("ax", "ay", "az", "gx", "gy", "gz", "timestamp")
    AX_FIELD_NUMBER: _ClassVar[int]
    AY_FIELD_NUMBER: _ClassVar[int]
    AZ_FIELD_NUMBER: _ClassVar[int]
    GX_FIELD_NUMBER: _ClassVar[int]
    GY_FIELD_NUMBER: _ClassVar[int]
    GZ_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    timestamp: int
    def __init__(self, ax: _Optional[float] = ..., ay: _Optional[float] = ..., az: _Optional[float] = ..., gx: _Optional[float] = ..., gy: _Optional[float] = ..., gz: _Optional[float] = ..., timestamp: _Optional[int] = ...) -> None: ...

class AdcDataRequest(_message.Message):
    __slots__ = ("channel",)
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    channel: int
    def __init__(self, channel: _Optional[int] = ...) -> None: ...

class AdcData(_message.Message):
    __slots__ = ("sample", "timestamp")
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    sample: float
    timestamp: int
    def __init__(self, sample: _Optional[float] = ..., timestamp: _Optional[int] = ...) -> None: ...

class RecordingEntry(_message.Message):
    __slots__ = ("scan", "imu")
    SCAN_FIELD_NUMBER: _ClassVar[int]
    IMU_FIELD_NUMBER: _ClassVar[int]
    scan: PointCloud3
    imu: IMUData
    def __init__(self, scan: _Optional[_Union[PointCloud3, _Mapping]] = ..., imu: _Optional[_Union[IMUData, _Mapping]] = ...) -> None: ...

class saveFileRequest(_message.Message):
    __slots__ = ("filename",)
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    filename: str
    def __init__(self, filename: _Optional[str] = ...) -> None: ...

class SensorStreamRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CameraStreamRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CameraStreamReply(_message.Message):
    __slots__ = ("width", "height", "encoding", "timestamp", "image_data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    encoding: CameraEncoding
    timestamp: int
    image_data: bytes
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., encoding: _Optional[_Union[CameraEncoding, str]] = ..., timestamp: _Optional[int] = ..., image_data: _Optional[bytes] = ...) -> None: ...
