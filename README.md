# 🖥️ MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents

## 📖 Introduction

Recent advances in large language models (LLMs) have significantly enhanced graphical user interface (GUI) agents. However, current benchmarks predominantly evaluate isolated tasks, leaving critical questions unanswered about cross-task correlations, performance-efficiency relationships, platform-specific differences, and primary bottlenecks.

MMBench-GUI, a hierarchical, multi-platform benchmarking framework, is introducted to address these gaps. MMBench-GUI is comprising four evaluation levels: GUI Content Understanding, GUI Element Grounding, GUI Task Automation, and GUI Task Collaboration. We also propose the Efficiency–Quality Area (EQA) metric, integrating accuracy and efficiency. MMBench-GUI provides a rigorous standard for evaluating and guiding future developments in GUI agent capabilities.

MMBench-GUI is developed based on [VLMEvalkit](https://github.com/open-compass/VLMEvalKit), supporting the evaluation of models in a API manner or local deployment manner. We hope that MMBench-GUI will enable more researchers to evaluate agents more efficiently and comprehensively. You can refer to the [How-to-Use](#how-to-use) section for usage details.

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

In this section, we provide detailed instructions on how to use MMBench-GUI to evaluate your model, including the design principle, the architecture of our code, the steps to adapt to your model, and some common issues we think. 

#### Design principle

We simply show the evaluation process in this figure. Our core **design principle** is to enable partial customization of model-side functionality based on currently common training/inference paradigms (such as `transformers` or API-based approaches), so that most of the code can be reused.

Specifically, we avoid introducing model files or external code directly into the project by supporting universal interfaces like `transformers.from_pretrained` for local deployment and `client.chat.completions.create` for OpenAI-style API calls.

Once the `data` flows out from the `dataloader`, the inference pipeline is divided into four stages:

1. `custom_build_prompt`: Organizes the `data` along with `system prompts` and `user prompts` into a standardized dictionary called `messages`.

2. `preprocess_func`: Further formats `messages` into `inputs` suitable for:

    - a processor (implemented via `transformers`) — yielding tokenized tensors like `input_ids`, or

    - a payload (for OpenAI-style APIs) — including the input content and corresponding generation configurations (e.g., temperature settings).

3. `generate`: Calls the model's `generate` method (if using `transformers`) or invokes the `client.chat.completions.create` (if using an OpenAI-style API) to produce predicted `outputs`.

4. `postprocess_func`: Processes the `outputs` into final `responses`, extracting the relevant content from the model's predictions.

After these steps, the evaluations of `responses` are conducted using `our_benchmark.evaluate`. In this process, all you need is to provide (or use our default implementation) a customized function to parse key informations.

**Based on above instruction, you are only required to implement a `custom_build_prompt`, a 'preprocess_func', a `preprocess_func`, and a `parse_response_func`. We have achieved a default function to handle these processes and if you do not have special format to be configured, you even DO NOT need to write these four functions!**

#### The architecture of our code

```text
MMBench-GUI/
|-- benchmarks                          // We put all levels of benchmarks here
|   |-- __init__.py
|   |-- l1_content_understanding.py
|   |-- l2_element_grounding.py
|   |-- l3_task_automation.py
|   |-- l4_task_collaboration.py
|   `-- matrics.py
|-- models                              // You are allowed to put code relative to your model here.
|   |-- __init__.py
|   |-- api_gpt.py
|   |-- api_uitars.py                   // One api-based example. We recommend you follow this example to implement your api-based model.
|   |-- base.py                         // Our model wrapper, including API-based and local-based.
|   |-- local_aguvis.py
|   `-- local_uitars.py                 // One local-based example. We recommend you follow this example to implement your local-based model.
|-- utils
|   |-- __init__.py
|   |-- download.py                     // You can run this to download and extract images and json files automatically.
|   |-- import_utils.py
|   |-- inference_tools.py              // Inference loops, including api-based and local-based.
|   `-- misc.py
|-- configs                             // You are allowed to put configs about your model here.
|   |-- config_api_gpt.json
|   |-- config_api_uitars.json          // One api-based config corresponding to models.api_uitars.py
|   |-- config_local_aguvis.json
|   `-- config_local_uitars.json        // One local-based config corresponding to models.local_uitars.py
|-- requirements
|   `-- dev_env.txt                     // We export the packages details of our development env.
|-- requirements.txt                    // You know
`-- evaluate.py                         // Run this file to start your evaluation.


```
