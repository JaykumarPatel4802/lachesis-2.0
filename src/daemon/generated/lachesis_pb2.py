# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lachesis.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0elachesis.proto\x12\x08lachesis\"i\n\x0fRegisterRequest\x12\x10\n\x08\x66unction\x18\x01 \x01(\t\x12\x15\n\rfunction_path\x18\x02 \x01(\t\x12\x19\n\x11\x66unction_metadata\x18\x03 \x03(\t\x12\x12\n\nparameters\x18\x04 \x03(\t\"g\n\rInvokeRequest\x12\x10\n\x08\x66unction\x18\x01 \x01(\t\x12\x0b\n\x03slo\x18\x02 \x01(\x01\x12\x12\n\nparameters\x18\x03 \x03(\t\x12\x13\n\x0b\x65xp_version\x18\x04 \x01(\t\x12\x0e\n\x06system\x18\x05 \x01(\t\"!\n\rDeleteRequest\x12\x10\n\x08\x66unction\x18\x01 \x01(\t\"\xb3\x02\n\x19InsertFunctionDataRequest\x12\n\n\x02pk\x18\x01 \x01(\x05\x12\x10\n\x08\x66unction\x18\x02 \x01(\t\x12\x12\n\nstart_time\x18\x03 \x01(\t\x12\x10\n\x08\x65nd_time\x18\x04 \x01(\t\x12\x10\n\x08\x64uration\x18\x05 \x01(\x01\x12\x1a\n\x12\x63old_start_latency\x18\x06 \x01(\x01\x12\x0f\n\x07p90_cpu\x18\x07 \x01(\x01\x12\x0f\n\x07p95_cpu\x18\x08 \x01(\x01\x12\x0f\n\x07p99_cpu\x18\t \x01(\x01\x12\x0f\n\x07max_cpu\x18\n \x01(\x01\x12\x0f\n\x07max_mem\x18\x0b \x01(\x01\x12\x12\n\ninvoker_ip\x18\x0c \x01(\t\x12\x14\n\x0cinvoker_name\x18\r \x01(\t\x12\x15\n\ractivation_id\x18\x0e \x01(\t\x12\x0e\n\x06\x65nergy\x18\x0f \x01(\x01\"(\n\x05Reply\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x0f\n\x07message\x18\x02 \x01(\t2\xfe\x01\n\x08Lachesis\x12\x38\n\x08Register\x12\x19.lachesis.RegisterRequest\x1a\x0f.lachesis.Reply\"\x00\x12\x34\n\x06Invoke\x12\x17.lachesis.InvokeRequest\x1a\x0f.lachesis.Reply\"\x00\x12\x34\n\x06\x44\x65lete\x12\x17.lachesis.DeleteRequest\x1a\x0f.lachesis.Reply\"\x00\x12L\n\x12InsertFunctionData\x12#.lachesis.InsertFunctionDataRequest\x1a\x0f.lachesis.Reply\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lachesis_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTERREQUEST']._serialized_start=28
  _globals['_REGISTERREQUEST']._serialized_end=133
  _globals['_INVOKEREQUEST']._serialized_start=135
  _globals['_INVOKEREQUEST']._serialized_end=238
  _globals['_DELETEREQUEST']._serialized_start=240
  _globals['_DELETEREQUEST']._serialized_end=273
  _globals['_INSERTFUNCTIONDATAREQUEST']._serialized_start=276
  _globals['_INSERTFUNCTIONDATAREQUEST']._serialized_end=583
  _globals['_REPLY']._serialized_start=585
  _globals['_REPLY']._serialized_end=625
  _globals['_LACHESIS']._serialized_start=628
  _globals['_LACHESIS']._serialized_end=882
# @@protoc_insertion_point(module_scope)
