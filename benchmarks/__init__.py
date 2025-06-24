from .l1_content_understanding import *
from .l2_element_grounding import *
from .l3_task_automation import *
from .l4_task_collaboration import *

BENCHMARK = {
    "GUIElementGrounding": GUIElementGrounding,
    "GUIContentUnderstanding": GUIContentUnderstanding,
    "GUITaskAutomation": GUITaskAutomation,
    "GUITaskCollaboration": GUITaskCollaboration,
}
