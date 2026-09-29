# Explainability methodology

The project treats an explanation as an object that should be evaluated rather than displayed alone.

## Integrated Gradients

Integrated Gradients attributes the prediction to input features relative to a baseline. It is implemented with Captum.

## Grad-CAM

Grad-CAM uses gradients flowing into a selected convolutional layer to create a spatial attribution map. The evaluator interpolates the map to the input image size.

## Attention visualization

The included TinyViT demo exposes token-level attention-related activation scores. For production research, the exact attention rollout protocol should be fixed before comparing models.

## Faithfulness

The deletion test progressively masks the most-attributed pixels and measures the reduction in model confidence. Larger confidence reduction can indicate that the attribution identified features important to the prediction, but the result depends on the masking strategy.

## Stability

The stability test compares attribution vectors before and after a small input perturbation using cosine similarity. It is a sensitivity analysis, not a causal validity guarantee.
