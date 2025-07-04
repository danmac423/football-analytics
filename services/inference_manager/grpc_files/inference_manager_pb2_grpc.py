# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import grpc

import services.inference_manager.grpc_files.inference_manager_pb2 as inference__manager__pb2

GRPC_GENERATED_VERSION = '1.69.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + ' but the generated code in inference_manager_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class InferenceManagerServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessFrames = channel.stream_stream(
                '/inference_manager.InferenceManagerService/ProcessFrames',
                request_serializer=inference__manager__pb2.Frame.SerializeToString,
                response_deserializer=inference__manager__pb2.Frame.FromString,
                _registered_method=True)


class InferenceManagerServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessFrames(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceManagerServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessFrames': grpc.stream_stream_rpc_method_handler(
                    servicer.ProcessFrames,
                    request_deserializer=inference__manager__pb2.Frame.FromString,
                    response_serializer=inference__manager__pb2.Frame.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'inference_manager.InferenceManagerService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('inference_manager.InferenceManagerService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class InferenceManagerService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessFrames(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/inference_manager.InferenceManagerService/ProcessFrames',
            inference__manager__pb2.Frame.SerializeToString,
            inference__manager__pb2.Frame.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
