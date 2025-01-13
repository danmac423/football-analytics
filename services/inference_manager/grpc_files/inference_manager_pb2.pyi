from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ("frame_id", "content")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    content: bytes
    def __init__(self, frame_id: _Optional[int] = ..., content: _Optional[bytes] = ...) -> None: ...
