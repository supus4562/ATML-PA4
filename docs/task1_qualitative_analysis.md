# Task 1: Qualitative Prompt-Level Evaluation

A prompt-level qualitative comparison across compressed model variants evaluated on zero-shot HellaSwag.

| ID | Phenomenon Analyzed | Context Prompt Excerpt | Gold Ending | Baseline | P1 (Mag) | P2 (Wanda) | 2:4 Struct | Q1 (PTQ) | Q2 (AWQ) | Q3 (Mixed) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #23 | Wanda vs Magnitude (50% Sparsity) | *"A group of men on donkeys play a form of polo on a dirt fiel..."* | *"camera pans to people watching the game."* | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| #41 | Wanda vs Magnitude (50% Sparsity) | *"One person makes a shot with a black puck. The next person m..."* | *"continue to alternate shots between a yellow ..."* | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| #23 | Unstructured Wanda vs 2:4 Structured (50% Sparsity) | *"A group of men on donkeys play a form of polo on a dirt fiel..."* | *"camera pans to people watching the game."* | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| #52 | Unstructured Wanda vs 2:4 Structured (50% Sparsity) | *"A little boy is standing eating ice cream. There is a dog on..."* | *"laughs and turns around."* | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| #6 | AWQ vs Uniform PTQ (INT4) | *"The family enjoys eating the desert together. The people in ..."* | *"gets up and walks away to the other room."* | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| #63 | AWQ vs Uniform PTQ (INT4) | *"He moves closer to the camera so that you can see the detail..."* | *"ends with the closing credits shown across th..."* | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

## Detailed Case Study Breakdown

### Case 1: Wanda vs Magnitude (50% Sparsity) (Sample #23)
- **Prompt Context:** "A group of men on donkeys play a form of polo on a dirt field. the"
- **Ground Truth Target:** "camera pans to people watching the game."
- **Theoretical Rationale:** P2 Wanda preserves large-magnitude activation channels, enabling correct reasoning where Magnitude pruning destroys critical outlier pathways.

### Case 2: Wanda vs Magnitude (50% Sparsity) (Sample #41)
- **Prompt Context:** "One person makes a shot with a black puck. The next person makes a shot with a yellow puck. the people"
- **Ground Truth Target:** "continue to alternate shots between a yellow and a black puck."
- **Theoretical Rationale:** P2 Wanda preserves large-magnitude activation channels, enabling correct reasoning where Magnitude pruning destroys critical outlier pathways.

### Case 3: Unstructured Wanda vs 2:4 Structured (50% Sparsity) (Sample #23)
- **Prompt Context:** "A group of men on donkeys play a form of polo on a dirt field. the"
- **Ground Truth Target:** "camera pans to people watching the game."
- **Theoretical Rationale:** The rigid 2-in-4 contiguous constraint forces the deletion of salient weights that cluster within the same 4-tuple, inducing catastrophic misclassification.

### Case 4: Unstructured Wanda vs 2:4 Structured (50% Sparsity) (Sample #52)
- **Prompt Context:** "A little boy is standing eating ice cream. There is a dog on the floor behind him. he"
- **Ground Truth Target:** "laughs and turns around."
- **Theoretical Rationale:** The rigid 2-in-4 contiguous constraint forces the deletion of salient weights that cluster within the same 4-tuple, inducing catastrophic misclassification.

### Case 5: AWQ vs Uniform PTQ (INT4) (Sample #6)
- **Prompt Context:** "The family enjoys eating the desert together. The people in the restaurant laugh at the man and he wonders what they are doing. the man"
- **Ground Truth Target:** "gets up and walks away to the other room."
- **Theoretical Rationale:** Channel-wise activation scaling under AWQ protects outlier features from aggressive clipping, preventing quantization-induced semantic collapse.

### Case 6: AWQ vs Uniform PTQ (INT4) (Sample #63)
- **Prompt Context:** "He moves closer to the camera so that you can see the detail of what he is doing with his hands. He begins to play different notes on the violin as he moves his fingers. the video"
- **Ground Truth Target:** "ends with the closing credits shown across the screen."
- **Theoretical Rationale:** Channel-wise activation scaling under AWQ protects outlier features from aggressive clipping, preventing quantization-induced semantic collapse.
