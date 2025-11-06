from rl.configs.config_plug import TrainConfig as PlugintoSocketwithPowerCordTrainConfig
from rl.configs.config_fake import TrainConfig as DebugTrainConfig
from rl.configs.config_open_switch import OpenSwitchTrainConfig


CONFIG_MAPPING = {
    "debug": DebugTrainConfig,
    "plug_into_socket_with_power_cord": PlugintoSocketwithPowerCordTrainConfig,
    "open_switch": OpenSwitchTrainConfig,
}
