# Style Transfer Tutorial

The simplest code for neural style transfer code using PyTorch.

| Content | Style | Result |
| --- | --- | --- |
| <img src="imgs/01.jpg" width="400px"> | <img src="imgs/candinsky.jpg" width="400px"> | <img src="output/01_candinsky/10000.png" width="400px"> |
| <img src="imgs/02.jpg" width="400px"> | <img src="imgs/candinsky.jpg" width="400px"> | <img src="output/02_candinsky/10000.png" width="400px"> |
| <img src="imgs/01.jpg" width="400px"> | <img src="imgs/cyberpunk02.jpg" width="400px"> | <img src="output/cyberpunk02_sgd/1000.png" width="400px"> |

## Run

```
python style_transfer.py
```

## Dependency

- Python 3
- torch==1.7
- pillow
- tqdm

## Reference

- https://junklee.tistory.com/69
- https://keras.io/examples/generative/neural_style_transfer/
- https://github.com/titu1994/Neural-Style-Transfer
