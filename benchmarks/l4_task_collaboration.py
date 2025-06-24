from vlmeval.dataset import ImageBaseDataset


class GUITaskCollaboration(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI_Task_Collaboration"
    DATASET_URL = {
        "MMBench-GUI_L4": "",  # noqa
    }  # path
    DATASET_MD5 = {
        "MMBench-GUI_L4": "",
    }
    RE_TYPE = "functional"

    # TODO: Coming soon
