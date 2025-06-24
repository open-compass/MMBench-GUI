from vlmeval.dataset import ImageBaseDataset


class GUITaskAutomation(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI_Task_Automation"
    DATASET_URL = {
        "MMBench-GUI_L3": "",  # noqa
    }  # path
    DATASET_MD5 = {
        "MMBench-GUI_L3": "",
    }
    RE_TYPE = "functional"

    # TODO: Coming soon
