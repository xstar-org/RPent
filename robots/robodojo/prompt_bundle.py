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

"""RoboDojo prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robodojo.prompts import evaluate as evaluate_parts
from robots.robodojo.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Return the RoboDojo system prompt.

    RoboDojo does not implement the exploration toolkit contract
    (``supports_exploration=False``), so every run mode uses the evaluation
    prompt.
    """
    del variables
    return {
        "ROLE": evaluate_parts.ROLE,
        "READ ORDER": evaluate_parts.READ_ORDER,
        "ACCURACY-FIRST LOOP": evaluate_parts.ACCURACY_LOOP,
        "TASK-FAMILY PLAYBOOKS": evaluate_parts.TASK_FAMILIES,
        "VLA AND PRIMITIVE CONTROL": evaluate_parts.CONTROL,
        "PERCEPTION": evaluate_parts.PERCEPTION,
        "RUNTIME": evaluate_parts.RUNTIME,
        "BUDGET AND SUCCESS": evaluate_parts.BUDGET_AND_SUCCESS,
        "MODE": evaluate_parts.USER_MODE,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    del variables
    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
