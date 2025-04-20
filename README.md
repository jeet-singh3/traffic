# Traffic Project

In this Traffic project, we created a convolutional neural network (CNN) using TensorFlow to classify road signs based
on images. The model was trained and evaluated using the German Traffic Sign Recognition Benchmark (GTSRB), a large
multi-category classification benchmark, each categorized by the type of sign it represents.

## Models

Each Model is implemented as a separate python function. Just replace the `return` statement on Line 318 to run.

Below is a summary of the different model architectures used in this project and what they were designed to test.

| Model Name                   | Description                                                                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `model_look_twice`           | Deeper Model. Adds a second layer. Looks at the image twice before making a decision.                                                     |
| `model_think_harder`         | Wider Model. Uses bigger filters and a larger dense layer. Thinking harder with more "neurons" might help it learn better.                |
| `model_remembers_everything` | No Dropout Model. Same as model_look_twice, but doesn’t forget anything.                                                                  |
| `model_try_minimum`          | Tiny Model. Very small and simple. Just one small conv and one small dense layer. Tests how little work the model can do and still learn. |
| `model_out_of_order`         | Out of Order Model. Dense layers come before the final conv layer. Testing weird stuff.                                                   |

## What We Learned

Through testing the variants, I gained several insights:

- **Deeper isn’t always better**: Adding a second convolutional layer (in `model_look_twice`) improved accuracy slightly at the cost of increased training time.
- **Wider models can overfit**: `model_think_harder` (larger filters and dense layers) showed strong training accuracy but overfit.
- **Dropout helps generalization**: `model_remembers_everything`, which removed dropout, trained quickly but performed worse on validation data.
- **Tiny models are surprising**: Despite its simplicity, `model_try_minimum` managed non-zero accuracy which I was not expecting.
- **Bad design hurts fast**: `model_out_of_order` performed terribly LOL
