from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ("frame_id", "content")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    content: bytes
    def __init__(
        self, frame_id: _Optional[int] = ..., content: _Optional[bytes] = ...
    ) -> None: ...

class PlayerInferenceResponse(_message.Message):
    __slots__ = ("frame_id", "boxes")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    BOXES_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    def __init__(
        self,
        frame_id: _Optional[int] = ...,
        boxes: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ...,
    ) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x1_n", "y1_n", "x2_n", "y2_n", "confidence", "class_label", "tracker_id")
    X1_N_FIELD_NUMBER: _ClassVar[int]
    Y1_N_FIELD_NUMBER: _ClassVar[int]
    X2_N_FIELD_NUMBER: _ClassVar[int]
    Y2_N_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    CLASS_LABEL_FIELD_NUMBER: _ClassVar[int]
    TRACKER_ID_FIELD_NUMBER: _ClassVar[int]
    x1_n: float
    y1_n: float
    x2_n: float
    y2_n: float
    confidence: float
    class_label: str
    tracker_id: int
    def __init__(
        self,
        x1_n: _Optional[float] = ...,
        y1_n: _Optional[float] = ...,
        x2_n: _Optional[float] = ...,
        y2_n: _Optional[float] = ...,
        confidence: _Optional[float] = ...,
        class_label: _Optional[str] = ...,
        tracker_id: _Optional[int] = ...,
    ) -> None: ...
