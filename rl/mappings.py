from rl.configs.config_plug import TrainConfig as PlugintoSocketwithPowerCordTrainConfig
from rl.configs.config_fake import TrainConfig as DebugTrainConfig


CONFIG_MAPPING = {
    "debug": DebugTrainConfig,
    "plug_into_socket_with_power_cord": PlugintoSocketwithPowerCordTrainConfig,
}
