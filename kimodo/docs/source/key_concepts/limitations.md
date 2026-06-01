# Best Practices

On this page, we summarize the best approaches to maximize Kimodo's capabilities in terms of prompting and constraints, and also summarize known limitations and failure cases. For additional context, please see the [tech report](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf).

## Text Prompting
- For best results, begin each prompt with "A person..." (e.g., "A person walks forward" or "A person jumps and waves"). This phrasing helps clarify the subject and intent of the motion, and is more closely aligned with the style of prompts used in the training data. The subject can also be stylized to better describe the motion such as "An old person..." or "A drunk person..."
- Keep each prompt focused one or at most two behaviors. For long sequences of action, split them into multiple prompts and generate in sequence.
- It's best to use a medium level of detail when describing a motion. Prompts like "A person walks." are too short and vague, while very long prompts describing detailed motion of each body part will be too much for the model to handle. Most training data is a middleground between these two. We recommend looking at the prompts in the [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed) to get an idea of prompt granularity.
- Kimodo is trained on a specific set of human behaviors. The training data tends to cover locomotion, gestures, everyday activities, common object interactions, videogame combat, dancing, and various styles including tired, angry, happy, sad, scared, drunk, injured, stealthy, old, and childlike. Prompts for actions outside of these categories will likely give bad results. For example "A baseball player walks up to the plate and swings a bat" is not good, becuase Kimodo has not trained on baseball data.
- When using multiple prompts (e.g., in the timeline UI), make sure each prompt has enough information on its own. For example, if prompt 1 is "A person is walking while carrying an object", then prompt 2 could be "A person walking carrying an object comes to a stop". If prompt 2 were instead "Then the person stops", the model will not have enough context for what happened previously and may generat poor quality motions.

## Constraints
- Avoid using constraints that contradict the given text prompt or other types of constraints. If you are having trouble with a tradeoff between constraint and text accuracy, try adjusting the [classifier-free guidance weights](../user_guide/configuration.md).
- Except for dense 2d root paths, Kimodo is mainly trained to handle sparse temporal constraints. Kimodo will perform best when the number of constraints per constraint type is less than 20 keyframes.
- When foot contact accuracy and hitting constraints is high priority, make sure to enable [post-processing](./constraints.md#post-processing).

## Limitations
- **Motion length:** Maximum generated motion duration is 10 sec per prompt
- **Number of constraints:** The number of constrained frames per constraint type should be less than 20 (excluding the root path constraint)
- **Overly long or complex prompts** can blur motion intent, especially when many distinct actions are packed into a single prompt.
- **Conflicting constraints:** can lead to artifacts or constraints that are ignored
- **Multi-prompt sequences**: When generating motions with a sequence of prompts, each motion is generated one at a time. The second motion is conditioned on the last frames of the first, so the transition between prompts actually happens at the start of the second motion. This means the second prompt must devote some of its duration to performing a smooth transition, which may reduce the time available to realize the new prompt content fully.
- **Post-processing**: The model by itself can generate foot skating and will not exactly hit constraints. Post-processing helps with this, but currently does not work well for the G1 robot skeleton.
