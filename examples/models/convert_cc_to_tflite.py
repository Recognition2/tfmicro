# Based on https://gist.github.com/petewarden/493294425ac522f00ff45342c71939d7
# Pete Warden 2020
# See https://petewarden.com/2020/02/28/converting-a-tensorflow-lite-cc-data-array-into-a-file-on-disk/

import re
import struct

def convert_cc_to_tflite(name, path):
    output_data = bytearray()
    with open(f"submodules/tensorflow/tensorflow/lite/micro/{path}", "r") as file:
        for line in file:
            values_match = re.match(r"\W*(0x[0-9a-fA-F,x ]+).*", line)
            if values_match:
                list_text = values_match.group(1)
                values_text = filter(None, list_text.split(","))
                values = [int(x, base=16) for x in values_text]
                output_data.extend(values)
    with open(f"examples/models/{name}", "wb") as output_file:
        output_file.write(output_data)

def convert_cc_to_tflite_float(name, path):
    output_data = bytearray()
    with open(f"submodules/tensorflow/tensorflow/lite/micro/{path}", "r") as file:
        for line in file:
            values_match = re.match(r"\s*([0-9,-. ]+).*", line)
            if values_match:
                list_text = values_match.group(1)
                values_text = filter(None, list_text.split(","))
                values = [b for x in values_text if len(x.strip()) for b in struct.pack("!f", float(x))]
                output_data.extend(values)
    with open(f"examples/models/{name}", "wb") as output_file:
        output_file.write(output_data)

def convert_cc_to_tflite_decimal(name, path, pack_format = "B"):
    output_data = bytearray()
    with open(f"submodules/tensorflow/tensorflow/lite/micro/{path}", "r") as file:
        for line in file:
            values_match = re.match(r"([0-9,\- ]+)", line)
            if values_match:
                list_text = values_match.group(1)
                values_text = filter(None, list_text.split(","))
                values = [int(x, base=10) for x in values_text if len(x.strip())]

                [output_data.extend(struct.pack(pack_format, v)) for v in values]
    with open(f"examples/models/{name}", "wb") as output_file:
        output_file.write(output_data)


# hello_world
convert_cc_to_tflite("hello_world.tflite", "examples/hello_world/model.cc")

# magic_wand
convert_cc_to_tflite(
    "magic_wand.tflite", "examples/magic_wand/magic_wand_model_data.cc"
)
convert_cc_to_tflite_float(
    "ring_micro_f9643d42_nohash_4.data",
    "examples/magic_wand/ring_micro_features_data.cc"
)
convert_cc_to_tflite_float(
    "slope_micro_f2e59fea_nohash_1.data",
     "examples/magic_wand/slope_micro_features_data.cc"
)

# micro_speech
convert_cc_to_tflite(
    "micro_speech.tflite", "examples/micro_speech/micro_features/model.cc"
)
convert_cc_to_tflite_decimal(
    "no_micro_f9643d42_nohash_4.data",
    "examples/micro_speech/micro_features/no_micro_features_data.cc",
)
convert_cc_to_tflite_decimal(
    "yes_micro_f2e59fea_nohash_1.data",
    "examples/micro_speech/micro_features/yes_micro_features_data.cc",
)

# micro_speech audio samples
convert_cc_to_tflite_decimal(
    "yes_1000ms_sample.data",
    "examples/micro_speech/yes_1000ms_sample_data.cc",
    "<h"
)
convert_cc_to_tflite_decimal(
    "no_1000ms_sample.data",
    "examples/micro_speech/no_1000ms_sample_data.cc",
    "<h"
)

# person_greyscale
convert_cc_to_tflite(
    "person_detection_grayscale.tflite",
    "tools/make/downloads/person_model_grayscale/person_detect_model_data.cc",
)
convert_cc_to_tflite(
    "person_image_data_grayscale.data",
    "tools/make/downloads/person_model_grayscale/person_image_data.cc",
)
convert_cc_to_tflite(
    "no_person_image_data_grayscale.data",
    "tools/make/downloads/person_model_grayscale/no_person_image_data.cc",
)
