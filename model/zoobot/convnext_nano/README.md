---
tags:
- image-classification
- timm
library_name: timm
license: apache-2.0
---
# Model card for zoobot-encoder-convnext_nano


Please see the [Zoobot docs](https://zoobot.readthedocs.io/en/latest/pretrained_models.html) for loading and finetuning instructions.

But minimally, you can use this like any timm encoder:

```python
import timm

encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-some-name', pretrained=True, num_classes=0)
```

