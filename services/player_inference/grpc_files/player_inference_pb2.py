# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: player_inference.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 29, 0, "", "player_inference.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x16player_inference.proto\x12\x15yolo.player_inference"*\n\x05\x46rame\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\x0c"^\n\x17PlayerInferenceResponse\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x05\x12\x31\n\x05\x62oxes\x18\x02 \x03(\x0b\x32".yolo.player_inference.BoundingBox"\x82\x01\n\x0b\x42oundingBox\x12\x0c\n\x04x1_n\x18\x01 \x01(\x02\x12\x0c\n\x04y1_n\x18\x02 \x01(\x02\x12\x0c\n\x04x2_n\x18\x03 \x01(\x02\x12\x0c\n\x04y2_n\x18\x04 \x01(\x02\x12\x12\n\nconfidence\x18\x05 \x01(\x02\x12\x13\n\x0b\x63lass_label\x18\x06 \x01(\t\x12\x12\n\ntracker_id\x18\x07 \x01(\x05\x32\x82\x01\n\x1aYOLOPlayerInferenceService\x12\x64\n\x10InferencePlayers\x12\x1c.yolo.player_inference.Frame\x1a..yolo.player_inference.PlayerInferenceResponse(\x01\x30\x01\x62\x06proto3'  # noqa
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "player_inference_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_FRAME"]._serialized_start = 49
    _globals["_FRAME"]._serialized_end = 91
    _globals["_PLAYERINFERENCERESPONSE"]._serialized_start = 93
    _globals["_PLAYERINFERENCERESPONSE"]._serialized_end = 187
    _globals["_BOUNDINGBOX"]._serialized_start = 190
    _globals["_BOUNDINGBOX"]._serialized_end = 320
    _globals["_YOLOPLAYERINFERENCESERVICE"]._serialized_start = 323
    _globals["_YOLOPLAYERINFERENCESERVICE"]._serialized_end = 453
# @@protoc_insertion_point(module_scope)
