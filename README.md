# 🖥️ MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents

## 📖 Introduction

Recent advances in large language models (LLMs) have significantly enhanced graphical user interface (GUI) agents. However, current benchmarks predominantly evaluate isolated tasks, leaving critical questions unanswered about cross-task correlations, performance-efficiency relationships, platform-specific differences, and primary bottlenecks. 

MMBench-GUI, a hierarchical, multi-platform benchmarking framework, is introducted to address these gaps. MMBench-GUI is comprising four evaluation levels: GUI Content Understanding, GUI Element Grounding, GUI Task Automation, and GUI Task Collaboration. We also propose the Efficiency–Quality Area (EQA) metric, integrating accuracy and efficiency. MMBench-GUI provides a rigorous standard for evaluating and guiding future developments in GUI agent capabilities.

MMBench-GUI is developed based on [VLMEvalkit](https://github.com/open-compass/VLMEvalKit), supporting the evaluation of models in a API manner or local deployment manner. We hope that MMBench-GUI will enable more researchers to evaluate agents more efficiently and comprehensively. You can refer to the How-to-Use section for usage details.

### Contributions

* **Hierarchical Evaluation**: Motivated by the use of levels L1~L5 in autonomous driving, we developed the hierarchical evaluation framework to systematically and comprehensively assess GUI agents' capabilities. Specifically, we organize the evaluation framework into four ascending levels, from **level 1**-GUI Content Understanding, **level 2**-GUI Element Grounding, **level 3**-GUI Task Automation (single-app) to **level 4**-GUI Task Collaboration (multi-app). Each level is associated with a set of tasks of increasing complexity, designed to test the Agent’s proficiency in progressively more demanding scenarios

* **Support multi-platform evaluation**: we establish a robust, multi-platform dataset encompassing diverse operating systems, such as Windows, macOS, Linux, iOS, Android, and Web interfaces, ensuring extensive coverage and relevance to real-world applications. 

* **A more human-aligned evaluation metric for planning**: We should not only assess whether an agent can complete a planning task, but also whether it can do so efficiently. In other words, we value both speed and quality. Therefore, we propose the Efficiency–Quality Area (EQA) metric that balances accuracy and efficiency, rewarding agents that achieve task objectives with minimal operational step, to replace  Success Rate (SR).

* **Manually reviewed and optimized online task setup**: We conducted a thorough review of existing online tasks and excluded those that could not be completed due to issues such as network or account restrictions. This refinement improves the rationality and feasibility of the evaluation data, enabling a more accurate assessment of the agent's capabilities.

* **More up-to-date evaluation data and more comprehensive task design**: In both Level 1 and Level 2 tasks, we collected, annotated, and processed additional evaluation data through a semi-automated workflow to better assess the agent’s localization and understanding capabilities. The task (or instruction) definitions are more focused, with explicit distinctions in difficulty levels. Moreover, the shared data sources across different task types enable a more natural evaluation of an agent’s performance across multiple dimensions. Overall, the benchmark comprises over 8,000 tasks spanning various operating platforms.

### Todos

* [ ] Release our technical reports where we have evaluated some GUI Agents on our benchmark.
* [ ] Support `circular` mode for the evaluation of `GUIContentUnderstanding`.
* [ ] Support `GUITaskAutomation` based on Docker for all platforms.
* [ ] Support `GUITaskCollaboration` based on Docker for all platforms.


## 🪧 News

* 2025.06.24 We have released the refactoring code for level1-GUI Content Understanding and level2-GUI Element Grounding tasks. Next, tasks of level3 and level4 will also be integrated into this codebase.

* 2025.06.24 We have released the images and json files used in level1-GUI Content Understanding and level2-GUI Element Grounding tasks. Resources of level3 and level4 will be release in the next one or two weeks.

## Performance

> **Note:** We are validating the final results again. Thus, performance of models shown in this table would change and we will update this as soon as possible.

#### 1. Performance on Level1 - GUI Content Understanding.
![](./assets/level1.png)

#### 2. Performance on Level2 - GUI Element Grounding.
![](./assets/level2.png)

#### 3. Performance on Level3 - GUI Task Automation.
![](./assets/level3.png)

#### 4. Performance on Level4 - GUI Task Collaboration.
![](./assets/level4.png)

## How-to-Use


