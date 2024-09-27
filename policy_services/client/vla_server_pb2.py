# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vla_server.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10vla_server.proto\x12\nvla_server\"h\n\x12\x43reateRobotRequest\x12)\n\nrobot_type\x18\x01 \x01(\x0e\x32\x15.vla_server.RobotType\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x12\n\nkeep_alive\x18\x03 \x01(\x05\"c\n\x13\x43reateRobotResponse\x12\x10\n\x08robot_id\x18\x01 \x01(\t\x12)\n\nerror_code\x18\x02 \x01(\x0e\x32\x15.vla_server.ErrorType\x12\x0f\n\x07message\x18\x03 \x01(\t\"]\n\x12ProprioceptionData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12+\n\x08\x65ncoding\x18\x02 \x01(\x0e\x32\x19.vla_server.StateEncoding\x12\x0c\n\x04\x64\x61ta\x18\x03 \x03(\x01\"9\n\tImageData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x10\n\x08\x65ncoding\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"\x7f\n\x0bObservation\x12%\n\x06images\x18\x01 \x03(\x0b\x32\x15.vla_server.ImageData\x12\x36\n\x0eproprioception\x18\x02 \x03(\x0b\x32\x1e.vla_server.ProprioceptionData\x12\x11\n\ttime_step\x18\x03 \x01(\x05\"\x93\x01\n\x08TaskInfo\x12!\n\x14language_instruction\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x35\n\x11image_instruction\x18\x02 \x01(\x0b\x32\x15.vla_server.ImageDataH\x01\x88\x01\x01\x42\x17\n\x15_language_instructionB\x14\n\x12_image_instruction\"r\n\x0cRobotRequest\x12\x10\n\x08robot_id\x18\x01 \x01(\t\x12,\n\x0bobservation\x18\x02 \x01(\x0b\x32\x17.vla_server.Observation\x12\"\n\x04task\x18\x03 \x01(\x0b\x32\x14.vla_server.TaskInfo\"\x1c\n\nActionData\x12\x0e\n\x06\x61\x63tion\x18\x01 \x03(\x01\"e\n\x0c\x41\x63tionResult\x12,\n\x08\x65ncoding\x18\x01 \x01(\x0e\x32\x1a.vla_server.ActionEncoding\x12\'\n\x07\x61\x63tions\x18\x02 \x03(\x0b\x32\x16.vla_server.ActionData\"\x97\x01\n\rRobotResponse\x12\x10\n\x08robot_id\x18\x04 \x01(\t\x12)\n\nerror_code\x18\x01 \x01(\x0e\x32\x15.vla_server.ErrorType\x12\x0f\n\x07message\x18\x02 \x01(\t\x12-\n\x06result\x18\x03 \x01(\x0b\x32\x18.vla_server.ActionResultH\x00\x88\x01\x01\x42\t\n\x07_result*=\n\tRobotType\x12\x11\n\rUNKNOWN_ROBOT\x10\x00\x12\r\n\tALOHA_TLR\x10\x01\x12\x0e\n\nALOHA_HFLR\x10\x02*{\n\rStateEncoding\x12\x16\n\x12UNKNOWN_STATE_TYPE\x10\x00\x12\r\n\tPOS_EULER\x10\x01\x12\x0c\n\x08POS_QUAT\x10\x02\x12\t\n\x05JOINT\x10\x03\x12\x12\n\x0eJOINT_BIMANUAL\x10\x04\x12\x16\n\x12POS_EULER_BIMANUAL\x10\x07*\xc8\x01\n\x0e\x41\x63tionEncoding\x12\x17\n\x13UNKNOWN_ACTION_TYPE\x10\x00\x12\x0b\n\x07\x45\x45\x46_POS\x10\x01\x12\r\n\tJOINT_POS\x10\x02\x12\x16\n\x12JOINT_POS_BIMANUAL\x10\x03\x12\n\n\x06\x45\x45\x46_R6\x10\x04\x12\x11\n\rJOINT_POS_ABS\x10\x05\x12\x1a\n\x16JOINT_POS_BIMANUAL_ABS\x10\x06\x12\x14\n\x10\x45\x45\x46_POS_BIMANUAL\x10\x07\x12\x18\n\x14\x45\x45\x46_POS_BIMANUAL_ABS\x10\x08*d\n\tErrorType\x12\x0b\n\x07SUCCESS\x10\x00\x12\x13\n\x0fROBOT_NOT_EXIST\x10\x01\x12\x17\n\x13ROBOT_TYPE_MISMATCH\x10\x02\x12\x1c\n\x18ROBOT_PARAMETER_MISMATCH\x10\x03\x32\xac\x01\n\x0cRobotService\x12N\n\x0b\x43reateRobot\x12\x1e.vla_server.CreateRobotRequest\x1a\x1f.vla_server.CreateRobotResponse\x12L\n\x13ProcessRobotRequest\x12\x18.vla_server.RobotRequest\x1a\x19.vla_server.RobotResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vla_server_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ROBOTTYPE']._serialized_start=1075
  _globals['_ROBOTTYPE']._serialized_end=1136
  _globals['_STATEENCODING']._serialized_start=1138
  _globals['_STATEENCODING']._serialized_end=1261
  _globals['_ACTIONENCODING']._serialized_start=1264
  _globals['_ACTIONENCODING']._serialized_end=1464
  _globals['_ERRORTYPE']._serialized_start=1466
  _globals['_ERRORTYPE']._serialized_end=1566
  _globals['_CREATEROBOTREQUEST']._serialized_start=32
  _globals['_CREATEROBOTREQUEST']._serialized_end=136
  _globals['_CREATEROBOTRESPONSE']._serialized_start=138
  _globals['_CREATEROBOTRESPONSE']._serialized_end=237
  _globals['_PROPRIOCEPTIONDATA']._serialized_start=239
  _globals['_PROPRIOCEPTIONDATA']._serialized_end=332
  _globals['_IMAGEDATA']._serialized_start=334
  _globals['_IMAGEDATA']._serialized_end=391
  _globals['_OBSERVATION']._serialized_start=393
  _globals['_OBSERVATION']._serialized_end=520
  _globals['_TASKINFO']._serialized_start=523
  _globals['_TASKINFO']._serialized_end=670
  _globals['_ROBOTREQUEST']._serialized_start=672
  _globals['_ROBOTREQUEST']._serialized_end=786
  _globals['_ACTIONDATA']._serialized_start=788
  _globals['_ACTIONDATA']._serialized_end=816
  _globals['_ACTIONRESULT']._serialized_start=818
  _globals['_ACTIONRESULT']._serialized_end=919
  _globals['_ROBOTRESPONSE']._serialized_start=922
  _globals['_ROBOTRESPONSE']._serialized_end=1073
  _globals['_ROBOTSERVICE']._serialized_start=1569
  _globals['_ROBOTSERVICE']._serialized_end=1741
# @@protoc_insertion_point(module_scope)
