# This program is used as compression demo
import json
from xtune.util import generate_configspace


def space():
    configurations = {
        "level": {
            "type": "int",
            "bound": [0, 9],
            "step": 1,
            "default": 6
        },
        "method": {
            "type": "cate",
            "choices": ["bz2", "zlib", "gzip", "lzma"],
            "default": "zlib"
        }
    }

    return generate_configspace(configurations)


def objective(configuration):
    with open("data.json", "r", encoding="utf-8") as fp:
        data = json.load(fp)
    method = configuration["method"]
    level = configuration["level"]
    compress_ratio, compress_time = data.get("%s-%d"%(method, level))
    return -compress_ratio * 1.0e6, compress_time * 60 * 60
