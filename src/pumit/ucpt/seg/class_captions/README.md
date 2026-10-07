# Class captions

Each JSON file defines the captions for one preprocessing `source`. The filename must equal `<source>.json`, and each document has this shape:

```json
{
  "source": "BraTS2023-PED",
  "classes": {
    "NC": [
      "non-enhancing tumor core"
    ]
  }
}
```

Class keys are the raw class names emitted by preprocessing. Description order is significant: the first description is the canonical text used without augmentation, while training uniformly samples the full list as text variants. Every description must accurately denote the same target given the image context.

Runtime code loads and validates this directory directly with [`load_class_captions()`](/src/pumit/text_prompt.py#L13). There is no generated aggregate caption file to rebuild.
