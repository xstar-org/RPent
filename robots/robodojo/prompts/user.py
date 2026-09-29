# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User prompt for one RoboDojo run."""

CELL = """- task: {{task_name}}
- seed: {{seed}}
- env_cfg: {{env_cfg}}
- embodiment: ARX X5 dual-arm (joint VLA + ee-pose primitives)
- checkpoint: RoboDojo-pi05-checkpoints sim-10task/14077
"""

BEGIN = """Follow the required read order, bind the current task's targets and
relations from fresh observation, then execute the first unmet phase. After
each action verify its observable gate, preserve achieved relations, and rely
on pi05_act with the native instruction for contact-rich phases."""
