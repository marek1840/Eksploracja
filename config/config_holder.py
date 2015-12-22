import json


class ConfigHolder(dict):
    __instance = None

    def __new__(cls):
        if ConfigHolder.__instance is None:
            ConfigHolder.__instance = dict.__new__(cls)
        return ConfigHolder.__instance

    def __init__(self, name=None, config_module_abs_path='conf.json'):
        with open(config_module_abs_path, 'r') as f:
            json_cfg = json.load(f)
        name = name or self.__class__.__name__.lower()
        dict.__init__(self, json_cfg)
        dict.__setattr__(self, "_name", name)

    def __str__(self):
        return repr(self)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        return super(ConfigHolder, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(ConfigHolder, self).__getitem__(name)

    def __delitem__(self, name):
        return super(ConfigHolder, self).__delitem__(name)
