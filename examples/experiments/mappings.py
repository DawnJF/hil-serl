# from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
# from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
# from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.usb_pickup_insertion.config_iphone import TrainConfig as IphoneChargingTrainConfig
from experiments.usb_pickup_insertion.config_plug import TrainConfig as PlugintoSocketwithPowerCordTrainConfig
from experiments.usb_pickup_insertion.config_fixed_pos import TrainConfig as PlugintoFixedSocketTrainConfig
from experiments.usb_pickup_insertion.config_flexible import TrainConfig as PlugintoMovableSocketTrainConfig


CONFIG_MAPPING = {
                # "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                # "object_handover": ObjectHandoverTrainConfig,
                # "egg_flip": EggFlipTrainConfig,
                "iphone_charging": IphoneChargingTrainConfig,
                "plug_into_socket_with_power_cord": PlugintoSocketwithPowerCordTrainConfig,
                "plug_into_fixed_socket": PlugintoFixedSocketTrainConfig,
                "plug_into_movable_socket": PlugintoMovableSocketTrainConfig,
                }